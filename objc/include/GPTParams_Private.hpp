#ifndef GPTParams_Private_hpp
#define GPTParams_Private_hpp

#import "GPTParams.h"
#import "ggml.h"
#import "../../common/common.h"


@interface GPTParams()

- (common_params&)params;

@end

@interface GPTSamplerParams()

- (common_params_sampling&)cParams;

@end
#endif /* GPTParams_Private_hpp */
