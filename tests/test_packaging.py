from unittest import TestCase
from inspect import ismodule, isfunction, getmembers

SUPV = {
    'lgx_rg_methods': [
        'cost_func',
        'g',
        'grad',
        'h',
        'p',
        'predict',
        'predict_prob',
        'reg_cost_func',
        'reg_grad'
    ],
    'lin_rg_methods': [
        'cost_func',
        'grad',
        'h',
        'inv',
        'normal_eqn',
        'predict',
        'reg_cost_func',
        'reg_grad'
    ],
    'nn_clsf_methods': [
        'append',
        'back_propagation',
        'cost_function',
        'dot',
        'feed_forward',
        'g',
        'g_grad',
        'grad',
        'h',
        'init_nn_weights',
        'ones',
        'rand_init_weights',
        'reshape',
        'sum',
        'unravel_params'
    ]
}


UNSUPV = {
    'kmeans_methods': [
        'compute_centroids',
        'cost_function',
        'elbow_method',
        'euclidean_dist',
        'find_closest_centroids',
        'init_centroids', 'mean',
        'run_intensive_kmeans',
        'run_kmeans',
        'sum'
    ],
    'pca_methods': [
        'diag',
        'pca',
        'project_data',
        'recover_data',
        'svd'
    ],
    'anmly_detc_methods': [
        'cov_matrix',
        'det',
        'estimate_multi_gaussian',
        'estimate_uni_gaussian',
        'inv',
        'is_anomaly',
        'mean',
        'multi_gaussian',
        'predict',
        'sum',
        'uni_gaussian',
        'var'
    ]
}


class TestPackaging(TestCase):
    def test_import_package(self):
        import touvlo

        assert ismodule(touvlo)

    def test_import_submodules_utils(self):
        from touvlo import utils

        assert ismodule(utils)

# SUPV

    def test_import_submodule_supv(self):
        from touvlo import supv

        assert ismodule(supv)

    def test_import_submodules_from_supv(self):
        from touvlo.supv import lgx_rg, lin_rg, nn_clsf

        lgx_rg_mtds = [method[0] for method in getmembers(lgx_rg, isfunction)]
        lin_rg_mtds = [method[0] for method in getmembers(lin_rg, isfunction)]
        nn_clsf_mtds = [method[0]
                        for method in getmembers(nn_clsf, isfunction)]

        assert lgx_rg_mtds == SUPV['lgx_rg_methods']
        assert lin_rg_mtds == SUPV['lin_rg_methods']
        assert nn_clsf_mtds == SUPV['nn_clsf_methods']

# UNSUPV

    def test_import_submodule_unsupv(self):
        from touvlo import unsupv

        assert ismodule(unsupv)

    def test_import_submodules_unsupv_methods(self):
        from touvlo.unsupv import pca, kmeans, anmly_detc

        pca_mtds = [method[0] for method in getmembers(pca, isfunction)]
        kmeans_mtds = [method[0] for method in getmembers(kmeans, isfunction)]
        anmly_detc_mtds = [method[0]
                           for method in getmembers(anmly_detc, isfunction)]

        assert pca_mtds == UNSUPV['pca_methods']
        assert kmeans_mtds == UNSUPV['kmeans_methods']
        assert anmly_detc_mtds == UNSUPV['anmly_detc_methods']
