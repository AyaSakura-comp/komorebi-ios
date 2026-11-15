// ane_rt.h — minimal direct Apple Neural Engine runtime (no CoreML).
// Compiles MIL text + weight blob in memory via the private AppleNeuralEngine.framework
// (_ANEInMemoryModelDescriptor / _ANEInMemoryModel / _ANERequest / _ANEIOSurfaceObject)
// and runs it with IOSurface-backed inputs/outputs.
// Approach adapted from github.com/maderix/ANE (MIT). Private API: not App Store safe.
#pragma once
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#include <stdint.h>
#include <stdbool.h>

NS_ASSUME_NONNULL_BEGIN

/// Builds a MIL weight.bin blob (64-byte file header + per-tensor 64-byte chunk headers, 64-byte aligned).
@interface ANEBlobBuilder : NSObject
/// Appends raw tensor bytes; returns the BLOBFILE offset (chunk header position) to use in MIL.
/// dtype codes: 1=fp16, 2=fp32, 3=uint8, 4=int8, 8=int4, 11=uint4.
- (uint64_t)append:(const void *)bytes length:(size_t)length dtype:(uint32_t)dtype;
/// Reserves space and returns a writable pointer to the payload (valid until the next append/finish).
- (uint64_t)appendZeroed:(size_t)length dtype:(uint32_t)dtype payload:(void * _Nullable * _Nonnull)payload;
- (NSData *)finish;
@property (nonatomic, readonly) size_t size;
@end

/// One compiled + loaded ANE program with one procedure per MIL function (in declaration order).
@interface ANEProgram : NSObject
+ (nullable instancetype)compileMIL:(NSString *)mil
                            weights:(nullable NSData *)weights
                         procedures:(NSArray<NSDictionary *> *)procs   // each: @{@"in": @[@bytes...], @"out": @[@bytes...]}
                              error:(NSError * _Nullable * _Nullable)error;
/// Synchronous evaluation of procedure `p`.
- (BOOL)eval:(int)p error:(NSError * _Nullable * _Nullable)error;
- (IOSurfaceRef)inputSurface:(int)i procedure:(int)p;
- (IOSurfaceRef)outputSurface:(int)i procedure:(int)p;
- (void *)inputPtr:(int)i procedure:(int)p;   // base address (surface stays CPU-mapped)
- (void *)outputPtr:(int)i procedure:(int)p;
/// Symbol names of the procedure's inputs / outputs in surface order (from the compiled model description).
- (NSArray<NSString *> *)inputNames:(int)p;
- (NSArray<NSString *> *)outputNames:(int)p;
/// Drops the descriptor's reference to the weight blob once the program is loaded on the ANE, so a memory-mapped blob
/// is unmapped when the caller lets go of it too (an iOS app without the extended virtual addressing entitlement has
/// only a few GB of address space). The loaded program keeps running; it cannot be reloaded from this object.
- (void)releaseWeights;
@property (nonatomic, readonly) double compileSeconds;
/// YES when aned already had the compiled program (compile was skipped).
@property (nonatomic, readonly) BOOL loadedFromCache;
@property (nonatomic, readonly) double loadSeconds;
@end

/// Number of in-memory compiles performed by this process.
int ane_rt_compile_count(void);
/// Seconds spent / bytes written staging programs to <tmp>/<model hash>/ for the ANE compiler (files already there
/// are reused).
double ane_rt_stage_seconds(void);
/// Seconds spent creating model descriptors and their identifiers (hashes the MIL and the weights).
double ane_rt_describe_seconds(void);
unsigned long long ane_rt_staged_bytes(void);
bool ane_rt_available(void);

NS_ASSUME_NONNULL_END
