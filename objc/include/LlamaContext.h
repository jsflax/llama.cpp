#ifndef LlamaContext_h
#define LlamaContext_h

@class GGMLThreadpool;
@class LlamaBatch;

typedef NSInteger LlamaSequenceId;
typedef NSInteger LlamaPosition;
typedef int32_t LlamaToken;
@class GPTParams;
@class LlamaModel;

typedef NS_ENUM(NSInteger, LlamaPoolingType) {
    LlamaPoolingTypeUnspecified = -1,
    LlamaPoolingTypeNone = 0,
    LlamaPoolingTypeMean = 1,
    LlamaPoolingTypeCls  = 2,
    LlamaPoolingTypeLast = 3,
    LlamaPoolingTypeRank = 4, // used by reranking models to attach the classification head to the graph
};

NS_ASSUME_NONNULL_BEGIN

@interface LlamaContext : NSObject

@property (nonatomic, assign, readonly) LlamaModel *model;
@property (nonatomic, assign) LlamaPoolingType poolingType;

- (instancetype _Nonnull)initWithParams:(GPTParams * _Nonnull)params;

- (NSUInteger)nCtx;

- (void)attachThreadpool:(GGMLThreadpool *)threadpool
         threadpoolBatch:(GGMLThreadpool *)threadpoolBatch;

// Positive return values does not mean a fatal error, but rather a warning.
//   0 - success
//   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
// < 0 - error
- (NSInteger)decode:(LlamaBatch *)batch;


-(void)decode:(LlamaBatch *)batch
output:(float *)output
nSeq:(int)nSeq
nEmbd:(int)nEmbd
embdNorm:(int)embdNorm;

// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
// If the KV cache is RoPEd, the KV data is updated accordingly:
//   - lazily on next llama_decode()
//   - explicitly with llama_kv_cache_update()
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
- (void)kvCacheSeqAdd:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta;

// Integer division of the positions by factor of `d > 1`
// If the KV cache is RoPEd, the KV data is updated accordingly:
//   - lazily on next llama_decode()
//   - explicitly with llama_kv_cache_update()
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
- (void)kvCacheSeqDiv:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta;

- (BOOL)kvCacheSeqRm:(LlamaSequenceId)sequenceId
                  p0:(LlamaPosition)p0
                  p1:(LlamaPosition)p1;

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
- (NSString *)tokenToPiece:(LlamaToken)token;
- (NSString *)tokenToPiece:(LlamaToken)token special:(BOOL)special;

- (BOOL)saveStateFile:(NSString *)pathSession
               tokens:(const LlamaToken *)tokens
          nTokenCount:(size_t)nTokenCount;

- (NSArray<NSNumber *> *)tokenize:(NSString *)text
                       addSpecial:(BOOL)addSpecial
                     parseSpecial:(BOOL)parseSpecial;

- (void)reset;

@end

NS_ASSUME_NONNULL_END

#endif /* LlamaContext_h */
