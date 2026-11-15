// ane_rt.m — see ane_rt.h
#import "ane_rt.h"
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <mach/mach_time.h>
#include <os/lock.h>

static Class gDesc, gInMem, gReq, gIO;
static os_unfair_lock gStatsLock = OS_UNFAIR_LOCK_INIT;
static double gStageSeconds = 0;
static double gDescribeSeconds = 0;
double ane_rt_describe_seconds(void) { return gDescribeSeconds; }
static unsigned long long gStagedBytes = 0;
double ane_rt_stage_seconds(void) { return gStageSeconds; }
unsigned long long ane_rt_staged_bytes(void) { return gStagedBytes; }
static int gCompileCount = 0;

static bool ane_load(void) {
    static dispatch_once_t once;
    static bool ok = false;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        gDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        gInMem = NSClassFromString(@"_ANEInMemoryModel");
        gReq   = NSClassFromString(@"_ANERequest");
        gIO    = NSClassFromString(@"_ANEIOSurfaceObject");
        ok = gDesc && gInMem && gReq && gIO;
    });
    return ok;
}

bool ane_rt_available(void) { return ane_load(); }
int ane_rt_compile_count(void) { return gCompileCount; }

static double now_s(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

#pragma mark - Blob

@implementation ANEBlobBuilder {
    NSMutableData *_d;
    uint32_t _count;
}
- (instancetype)init {
    if ((self = [super init])) {
        _d = [NSMutableData dataWithLength:64];
    }
    return self;
}
- (size_t)size { return _d.length; }
- (uint64_t)appendZeroed:(size_t)length dtype:(uint32_t)dtype payload:(void **)payload {
    size_t hdr = _d.length;            // always 64-aligned
    size_t dataOff = hdr + 64;
    size_t padded = (length + 63) & ~(size_t)63;
    [_d increaseLengthBy:64 + padded];
    uint8_t *h = (uint8_t *)_d.mutableBytes + hdr;
    *(uint32_t *)(h + 0) = 0xDEADBEEF;
    *(uint32_t *)(h + 4) = dtype;
    *(uint64_t *)(h + 8) = (uint64_t)length;
    *(uint64_t *)(h + 16) = (uint64_t)dataOff;
    _count++;
    *payload = (uint8_t *)_d.mutableBytes + dataOff;
    return hdr;
}
- (uint64_t)append:(const void *)bytes length:(size_t)length dtype:(uint32_t)dtype {
    void *p = NULL;
    uint64_t off = [self appendZeroed:length dtype:dtype payload:&p];
    memcpy(p, bytes, length);
    return off;
}
- (NSData *)finish {
    uint8_t *g = (uint8_t *)_d.mutableBytes;
    *(uint32_t *)(g + 0) = _count;
    *(uint32_t *)(g + 4) = 2;
    return _d;
}
@end

#pragma mark - Program

@interface ANEProc : NSObject
@property (nonatomic, strong) id request;
@property (nonatomic, assign) IOSurfaceRef *ins;
@property (nonatomic, assign) IOSurfaceRef *outs;
@property (nonatomic, assign) void **inPtrs;
@property (nonatomic, assign) void **outPtrs;
@property (nonatomic, assign) int nIn, nOut;
@property (nonatomic, strong) NSArray<NSString *> *inNames;
@property (nonatomic, strong) NSArray<NSString *> *outNames;
@end
@implementation ANEProc
- (void)dealloc {
    for (int i = 0; i < _nIn; i++) CFRelease(_ins[i]);
    for (int i = 0; i < _nOut; i++) CFRelease(_outs[i]);
    free(_ins); free(_outs); free(_inPtrs); free(_outPtrs);
}
@end

static IOSurfaceRef make_surface(size_t bytes, void **ptr) {
    if (bytes < 64) bytes = 64;
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
    IOSurfaceLock(s, 0, NULL);
    *ptr = IOSurfaceGetBaseAddress(s);
    memset(*ptr, 0, bytes);
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

@implementation ANEProgram {
    id _model;
    BOOL _wasCached;
    NSString *_tmpDir;
    NSArray<ANEProc *> *_procs;
}

+ (instancetype)compileMIL:(NSString *)mil weights:(NSData *)weights procedures:(NSArray<NSDictionary *> *)procs error:(NSError **)error {
    if (!ane_load()) {
        if (error) *error = [NSError errorWithDomain:@"ANE" code:-1 userInfo:@{NSLocalizedDescriptionKey: @"AppleNeuralEngine.framework unavailable"}];
        return nil;
    }
    ANEProgram *p = [ANEProgram new];
    double td0 = now_s();
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *wdict = weights ? @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weights}} : @{};
    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(gDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
    if (!desc) {
        if (error) *error = [NSError errorWithDomain:@"ANE" code:-2 userInfo:@{NSLocalizedDescriptionKey: @"descriptor creation failed"}];
        return nil;
    }
    id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(gInMem, @selector(inMemoryModelWithDescriptor:), desc);
    NSString *hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    double describeDt = now_s() - td0;
    os_unfair_lock_lock(&gStatsLock); gDescribeSeconds += describeDt; os_unfair_lock_unlock(&gStatsLock);
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = NSFileManager.defaultManager;
    // The compiler reads the program from <tmp>/<model hash>/. The name is a hash of the MIL and the weights, so
    // files already there are identical: write only what is missing (the weights are ~1.1 GB per engine).
    void (^stageFiles)(void) = ^{
        double tw = now_s();
        NSString *milPath = [td stringByAppendingPathComponent:@"model.mil"];
        NSString *wPath = [td stringByAppendingPathComponent:@"weights/weight.bin"];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        if ([[fm attributesOfItemAtPath:milPath error:nil] fileSize] != milData.length) {
            [milData writeToFile:milPath atomically:NO];
            os_unfair_lock_lock(&gStatsLock); gStagedBytes += milData.length; os_unfair_lock_unlock(&gStatsLock);
        }
        if (weights && [[fm attributesOfItemAtPath:wPath error:nil] fileSize] != weights.length) {
            [weights writeToFile:wPath atomically:NO];
            os_unfair_lock_lock(&gStatsLock); gStagedBytes += weights.length; os_unfair_lock_unlock(&gStatsLock);
        }
        double dt = now_s() - tw;
        os_unfair_lock_lock(&gStatsLock); gStageSeconds += dt; os_unfair_lock_unlock(&gStatsLock);
    };

    NSError *e = nil;
    double t0 = now_s();
    // aned keeps compiled programs keyed by the model hash: reuse them instead of recompiling every launch.
    BOOL cached = NO;
    if ([mdl respondsToSelector:@selector(compiledModelExists)]) {
        cached = ((BOOL(*)(id, SEL))objc_msgSend)(mdl, @selector(compiledModelExists));
    }
    BOOL ok = YES;
    if (!cached) {
        stageFiles();
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        os_unfair_lock_lock(&gStatsLock); gCompileCount++; os_unfair_lock_unlock(&gStatsLock);
    }
    double t1 = now_s();
    if (!ok) { [fm removeItemAtPath:td error:nil]; if (error) *error = e; return nil; }
    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok && cached) {
        // Loading straight from aned's cache did not work: stage the files, then load (compile if still needed).
        e = nil;
        stageFiles();
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            e = nil;
            ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            os_unfair_lock_lock(&gStatsLock); gCompileCount++; os_unfair_lock_unlock(&gStatsLock);
            if (ok) ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        }
    }
    double t2 = now_s();
    if (!ok) { [fm removeItemAtPath:td error:nil]; if (error) *error = e; return nil; }
    p->_wasCached = cached;
    if (getenv("ANE_RT_DEBUG")) {
        id attrs = nil;
        @try { attrs = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(modelAttributes)); } @catch (id ex) {}
        NSLog(@"ANE model attributes: %@", attrs);
    }
    p->_compileSeconds = t1 - t0;
    p->_loadSeconds = t2 - t1;
    p->_model = mdl;
    p->_tmpDir = td;

    // Global symbol indices per procedure (multi-function programs number symbols across all functions).
    NSArray *procInfo = nil, *inSymAll = nil, *outSymAll = nil;
    @try {
        NSDictionary *attrs = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(modelAttributes));
        procInfo = attrs[@"ANEFModelDescription"][@"ANEFModelProcedures"];
        inSymAll = attrs[@"ANEFModelDescription"][@"kANEFModelInputSymbolsArrayKey"];
        outSymAll = attrs[@"ANEFModelDescription"][@"kANEFModelOutputSymbolsArrayKey"];
    } @catch (id ex) {}
    NSUInteger symIn = 0, symOut = 0;
    NSMutableArray *list = [NSMutableArray array];
    for (NSUInteger pi = 0; pi < procs.count; pi++) {
        NSArray *inSyms = nil, *outSyms = nil;
        if (pi < procInfo.count) {
            inSyms = procInfo[pi][@"ANEFModelInputSymbolIndexArray"];
            outSyms = procInfo[pi][@"ANEFModelOutputSymbolIndexArray"];
        }
        NSArray *ins = procs[pi][@"in"], *outs = procs[pi][@"out"];
        ANEProc *pr = [ANEProc new];
        pr.nIn = (int)ins.count; pr.nOut = (int)outs.count;
        pr.ins = calloc(ins.count, sizeof(IOSurfaceRef)); pr.outs = calloc(outs.count, sizeof(IOSurfaceRef));
        pr.inPtrs = calloc(ins.count, sizeof(void *)); pr.outPtrs = calloc(outs.count, sizeof(void *));
        NSMutableArray *wi = [NSMutableArray array], *ii = [NSMutableArray array];
        NSMutableArray *wo = [NSMutableArray array], *oi = [NSMutableArray array];
        for (NSUInteger i = 0; i < ins.count; i++) {
            pr.ins[i] = make_surface([ins[i] unsignedLongValue], &pr.inPtrs[i]);
            [wi addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gIO, @selector(objectWithIOSurface:), pr.ins[i])];
            [ii addObject:(i < inSyms.count) ? inSyms[i] : @(symIn + i)];
        }
        for (NSUInteger i = 0; i < outs.count; i++) {
            pr.outs[i] = make_surface([outs[i] unsignedLongValue], &pr.outPtrs[i]);
            [wo addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(gIO, @selector(objectWithIOSurface:), pr.outs[i])];
            [oi addObject:(i < outSyms.count) ? outSyms[i] : @(symOut + i)];
        }
        symIn += ins.count; symOut += outs.count;
        NSMutableArray *inN = [NSMutableArray array], *outN = [NSMutableArray array];
        for (NSNumber *n in ii) [inN addObject:(n.unsignedIntegerValue < inSymAll.count) ? inSymAll[n.unsignedIntegerValue] : @"?"];
        for (NSNumber *n in oi) [outN addObject:(n.unsignedIntegerValue < outSymAll.count) ? outSymAll[n.unsignedIntegerValue] : @"?"];
        pr.inNames = inN; pr.outNames = outN;
        pr.request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(gReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wi, ii, wo, oi, nil, nil, @(pi));
        [list addObject:pr];
    }
    p->_procs = list;
    return p;
}

- (BOOL)loadedFromCache { return _wasCached; }

- (void)releaseWeights {
    if (!_model) return;
    Ivar descIvar = class_getInstanceVariable(object_getClass(_model), "_descriptor");
    id desc = descIvar ? object_getIvar(_model, descIvar) : nil;
    Ivar wIvar = desc ? class_getInstanceVariable(object_getClass(desc), "_weights") : NULL;
    if (wIvar) object_setIvar(desc, wIvar, @{});
}

- (BOOL)eval:(int)pi error:(NSError **)error {
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(_model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, _procs[pi].request, &e);
    if (!ok && error) *error = e;
    return ok;
}

- (NSArray<NSString *> *)inputNames:(int)p { return _procs[p].inNames; }
- (NSArray<NSString *> *)outputNames:(int)p { return _procs[p].outNames; }
- (IOSurfaceRef)inputSurface:(int)i procedure:(int)p { return _procs[p].ins[i]; }
- (IOSurfaceRef)outputSurface:(int)i procedure:(int)p { return _procs[p].outs[i]; }
- (void *)inputPtr:(int)i procedure:(int)p { return _procs[p].inPtrs[i]; }
- (void *)outputPtr:(int)i procedure:(int)p { return _procs[p].outPtrs[i]; }

- (void)dealloc {
    if (_model) {
        NSError *e = nil;
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(_model, @selector(unloadWithQoS:error:), 21, &e);
    }
    if (_tmpDir) [NSFileManager.defaultManager removeItemAtPath:_tmpDir error:nil];
}
@end
