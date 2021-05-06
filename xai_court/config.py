'''Define configuration classes'''

class Config:
    '''Defines the parameters used across the code base which can be adjusted without modifying code directly'''
    logger_name = 'court-of-xai'
    package_name = 'xai_court'

    # 
    serialization_base_dir = 'outputs'

    # 
    seeds = [87, 2134, 5555]

    #
    mpl_style = 'seaborn-poster'

    #
    sns_palette = 'cubehelix'
