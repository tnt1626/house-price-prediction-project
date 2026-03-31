from cache_loading import MLflowTrainingCacheManager
from domain_feature import add_domain_features
from reprocessing import make_preprocessor, make_feature_space

cached = MLflowTrainingCacheManager()
pipeline = make_feature_space(df_train, df_test)
domain_feature = add_domain_features(data)