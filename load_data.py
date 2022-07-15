import os
from anndata import AnnData
import pandas as pd
import numpy as np
import scanpy as sc
import anndata  as ad
from scipy.sparse import coo_matrix
import umap

def convert_string_to_encoding(string, vector_key):
    """A function to convert a string to a numeric encoding.


    Arguments:
    ------------------------------------------------------------------
    - string: `str`, The specific string to convert to a numeric encoding.
    - vector_key: `np.ndarray`, Array of all possible values of string.

    Returns:
    ------------------------------------------------------------------
    - encoding: `int`, The integer encoding of string.
    """

    return np.argwhere(vector_key == string)[0][0]

def convert_vector_to_encoding(vector):
    """A function to convert a vector of strings to a dense numeric encoding.


    Arguments:
    ------------------------------------------------------------------
    - vector: `array_like`, The vector of strings to encode.

    Returns:
    ------------------------------------------------------------------
    - vector_num: `list`, A list containing the dense numeric encoding.
    """

    vector_key = np.unique(vector)
    vector_strings = list(vector)
    vector_num = [convert_string_to_encoding(string, vector_key) for string in vector_strings]

    return vector_num


def normalize_scanpy(adata, batch_key=None, n_high_var=2000, LVG=True,
                     normalize_samples=True, log_normalize=True,
                     normalize_features=False, origin_batch=None):
    """ This function preprocesses the raw count data.
    Arguments:
    ------------------------------------------------------------------
    - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.
    - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
    - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 2000, then the 2000 genes with the highest variance are designated as highly variable.
    - LVG: `bool`, Whether to retain and preprocess LVGs.
    - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
    - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
    - normalize_features: `bool`, If True, z-score normalize each gene's expression.
    Returns:
    ------------------------------------------------------------------
    - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Contains preprocessed data.
    """

    from scipy.sparse import issparse
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    n, p = adata.shape
    sparsemode = issparse(adata.X)

    # convert batch string into number
    if batch_key is not None:
        batch = list(adata.obs[batch_key])
        batch_origin=batch
        batch = convert_vector_to_encoding(batch)
        batch = np.asarray(batch)
        batch = batch.astype('float32')
    else:
        batch = np.ones((n,), dtype='float32')
        norm_by_batch = False


    count = adata.X.copy()

    if normalize_samples:
        out = sc.pp.normalize_total(adata, inplace=False)
        obs_ = adata.obs
        var_ = adata.var
        adata = None
        adata = AnnData(out['X'])
        adata.obs = obs_
        adata.var = var_

        size_factors = out['norm_factor'] / np.median(out['norm_factor'])
        out = None
    else:
        size_factors = np.ones((adata.shape[0],))

    if not log_normalize:
        adata_ = adata.copy()

    sc.pp.log1p(adata)

    if n_high_var is not None:
        sc.pp.highly_variable_genes(adata, inplace=True, min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    n_bins=20, n_top_genes=n_high_var, batch_key=batch_key)

        hvg = adata.var['highly_variable'].values

        if not log_normalize:
            adata = adata_.copy()

    else:
        hvg = [True] * adata.shape[1]

    if normalize_features:
        batch_list = np.unique(batch)

        if sparsemode:
            adata.X = adata.X.toarray()

        for batch_ in batch_list:
            indices = [x == batch_ for x in batch]
            sub_adata = adata[indices]

            sc.pp.scale(sub_adata)
            #minmax_scale(sub_adata.X, feature_range=(0, 1), axis=0, copy=False)

            adata[indices] = sub_adata.X

        adata.layers["normalized input"] = adata.X
        adata.X = count
        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]

    else:
        if sparsemode:
            adata.layers["normalized input"] = adata.X.toarray()
        else:
            adata.layers["normalized input"] = adata.X

        adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]

    if n_high_var is not None:
        del_keys = ['dispersions', 'dispersions_norm', 'highly_variable', 'highly_variable_intersection',
                    'highly_variable_nbatches', 'means']
        del_keys = [x for x in del_keys if x in adata.var.keys()]
        adata.var = adata.var.drop(del_keys, axis=1)

    y = np.unique(batch)
    num_batch = len(y)

    adata.obs['size factors'] = size_factors.astype('float32')
    adata.obs['batch'] = batch
    adata.uns['num_batch'] = num_batch
    if origin_batch is not None:
        adata.obs['batch_origin']=origin_batch

    if sparsemode:
        adata.X = adata.X.toarray()

    if not LVG:
        adata = adata[:, adata.var['Variance Type'] == 'HVG']

    return adata


def convert_labels_to_digital(num_class_list, train_labels, test_labels):
    rename = {}
    unique_class = np.unique(np.array(num_class_list))
    unique_class.sort()
    for line in range(0, len(unique_class)):
        key = unique_class[line]
        rename[key] = int(line)
    print(rename)

    for index, labels in enumerate(train_labels):
        train_labels[index] = pd.DataFrame(train_labels[index]).replace(rename).values.flatten()
    if len(test_labels)>0:
        for index, labels in enumerate(test_labels):
            test_labels[index] = pd.DataFrame(test_labels[index]).replace(rename).values.flatten()

    return train_labels, test_labels,rename


def get_common_gene(name_list):
    root_path="/home/zengys/data/hd5_data/"
    common_gene=[]
    for name in name_list:
        data = sc.read_h5ad(root_path + name + "_count_data.h5ad")
        genes=list(data.var_names.values)
        if len(common_gene)==0:
            common_gene.extend(list(genes))
        else:
            common_gene=list(set(common_gene).intersection(set(genes)))

    return common_gene


def remove_clean_cell_types(dataset):
    dataset=dataset[dataset.obs["CellType"] !='Unassigned']

    # remove the number of cell type less than 10
    cell_types_list=dataset.obs['CellType'].values
    for cell_type in set(cell_types_list):
        if len(cell_types_list[cell_types_list==cell_type]) < 10:
            dataset = dataset[dataset.obs["CellType"] != cell_type]

    return dataset


def remove_clean_cell_types_pancreas(dataset):
    dataset = dataset[dataset.obs["orig.ident"] != 'unclear']
    dataset = dataset[dataset.obs["orig.ident"] != 'not applicable']
    dataset = dataset[dataset.obs["orig.ident"] != 'unclassified']
    dataset = dataset[dataset.obs["orig.ident"] != 'unclassified endocrine']

    dataset = dataset[dataset.obs["orig.ident"] != 'alpha.contaminated']
    dataset = dataset[dataset.obs["orig.ident"] != 'beta.contaminated']
    dataset = dataset[dataset.obs["orig.ident"] != 'delta.contaminated']
    dataset = dataset[dataset.obs["orig.ident"] != 'gamma.contaminated']



    # remove the number of cell type less than 10
    cell_types_list=dataset.obs['orig.ident'].values
    for cell_type in set(cell_types_list):
        if len(cell_types_list[cell_types_list==cell_type]) < 10:
            dataset = dataset[dataset.obs["orig.ident"] != cell_type]

    return dataset


def get_ranked_common_gene(data_names):
    root_path="/home/zengys/data/hd5_data/"
    norm_data_list = []
    common_gene = []
    ori_data_list=[]

    for i, dataset in enumerate(data_names):
        data_frame = sc.read_h5ad(root_path + dataset + "_count_data.h5ad")
        data_frame = remove_clean_cell_types(data_frame)
        data_frame.obs['batch']=[dataset]*data_frame.shape[0]

        ori_data_list.append(data_frame)

    combined_data=ad.concat(ori_data_list, join="outer")

    dataset = normalize_scanpy(combined_data, batch_key="batch")

    for i, name in enumerate(data_names):
        norm_data_list.append(dataset[dataset.obs.batch_origin==name][:, dataset.var['Variance Type'] == 'HVG'])

    return common_gene, norm_data_list



def get_combined_norm(data_names):
    root_path="/home/zengys/data/hd5_data/"
    norm_data_list = []
    common_gene = []
    ori_data_list=[]


    for i, dataset in enumerate(data_names):
        data_frame = sc.read_h5ad(root_path + dataset + "_count_data.h5ad")
        data_frame = remove_clean_cell_types(data_frame)
        data_frame.obs['batch']=[dataset]*data_frame.shape[0]

        ori_data_list.append(data_frame)

    combined_data=ad.concat(ori_data_list, join="outer")

    dataset = normalize_scanpy(combined_data, origin_batch=combined_data.obs['batch'].values, n_high_var=None)

    HVG=False
    if HVG:
        for i, name in enumerate(data_names):
            norm_data_list.append(dataset[dataset.obs.batch_origin==name][:, dataset.var['Variance Type'] == 'HVG'])
    else:
        for i, name in enumerate(data_names):
            norm_data_list.append(dataset[dataset.obs.batch_origin==name])

    return common_gene, norm_data_list


def get_norm_seperate(data_names,hvg=None,suffix="_count_data"):
    root_path="data/"
    norm_data_list = []
    common_gene = []
    ori_data_list=[]


    for i, dataset in enumerate(data_names):
        data_frame = sc.read_h5ad(root_path + dataset + suffix + ".h5ad")
        data_frame = remove_clean_cell_types(data_frame)
        dataset = normalize_scanpy(data_frame,  n_high_var=hvg)
        norm_data_list.append(dataset)
        genes = list(dataset.var_names.values)
        if len(common_gene) == 0:
            common_gene.extend(list(genes))
        else:
            if data_names[i]=='Muraro':
                genes=[gene.split("--")[0] for gene in genes]
                norm_data_list[i].var_names=genes
            common_gene = list(set(common_gene).intersection(set(genes)))

    for index,  data_set in enumerate(norm_data_list):
        norm_data_list[index]=norm_data_list[index][:, common_gene]

    return common_gene, norm_data_list


def get_norm_seperate_pancreas(data_names,hvg=None,suffix="_count_data"):#_count_data _seurat_norm
    root_path="data/"
    norm_data_list = []
    common_gene = []
    ori_data_list=[]


    for i, dataset in enumerate(data_names):
        print(dataset)
        data_frame = sc.read_h5ad(root_path + dataset + suffix + ".h5ad")
        data_frame = remove_clean_cell_types_pancreas(data_frame)
        dataset = normalize_scanpy(data_frame,  n_high_var=hvg)
        norm_data_list.append(dataset)
        genes = list(dataset.var_names.values)

        if len(common_gene) == 0:
            if data_names[i]=='Muraro':
                genes=[gene.split("--")[0] for gene in genes]
                norm_data_list[i].var_names=genes
            common_gene.extend(list(genes))
        else:
            if data_names[i]=='Muraro':
                genes=[gene.split("--")[0] for gene in genes]
                norm_data_list[i].var_names=genes
            common_gene = list(set(common_gene).intersection(set(genes)))

    for index,  data_set in enumerate(norm_data_list):
        norm_data_list[index]=norm_data_list[index][:, common_gene]

    return common_gene, norm_data_list


def load_numpy_data(name,
                    logger,
                    train_ratio=0.8,
                    hvg=None,
                    norm_combined=True):


    if name == "pbmc_ding":
        data_names = [
            "pbmcsca_10x_Chromium_v2",
            "pbmcsca_10x_Chromium_v3",
            "pbmcsca_CEL-Seq2",
            "pbmcsca_Seq-Well",
            "pbmcsca_Smart-seq2",
            "pbmcsca_Drop-seq",
            "pbmcsca_inDrops",
        ]
        key='CellType'
        norm=get_norm_seperate
    elif name == "pancreas":
        data_names = [
            "Baron_Mouse",
            "Baron_Human",
            "Xin",
            "Muraro",
            "Segerstolpe",
        ]
        key='orig.ident'
        norm=get_norm_seperate_pancreas
    else:
        raise ValueError("Unknown dataset, please add the new datasets into load data function")
    logger.info("data_name =%s" % data_names)
    if norm_combined:
        common_gene, norm_data_list=norm(data_names,hvg)
    else:
        common_gene, norm_data_list = get_ranked_common_gene(data_names)

    num_class_list = []
    train_insts, train_labels, test_insts, test_labels = [], [], [], []
    for i, dataset in enumerate(norm_data_list):

        data = dataset.X
        labels = dataset.obs[key].values
        print(list(np.unique(labels)))
        num_class_list.extend(np.unique(list(labels)))

        logger.info("data name: %s,  data shape %s" %(data_names[i], data.shape))
        train_insts.append(data)
        train_labels.append(np.squeeze(labels))

        # Random shuffle
        if train_ratio<1:
            num_trains = int(labels.shape[0] * train_ratio)
            ridx = np.arange(labels.shape[0])
            np.random.shuffle(ridx)
            test_insts.append(train_insts[i][ridx[num_trains:]])
            test_labels.append(train_labels[i][ridx[num_trains:]].ravel())
            train_insts[i] = train_insts[i][ridx[:num_trains]]
            train_labels[i] = train_labels[i][ridx[:num_trains]].ravel()

    train_labels, test_labels,rename=convert_labels_to_digital(num_class_list, train_labels, test_labels)

    configs = {"input_dim": data.shape[1],
               "hidden_layers": [1024, 512, 32],
               "num_classes": len(set(num_class_list)),
               "drop_rate": 0}

    return data_names, train_insts, train_labels, test_insts, test_labels, configs,rename


def data_loader(inputs, targets, batch_size, shuffle=True):
    assert inputs.shape[0] == targets.shape[0]
    inputs_size = inputs.shape[0]
    if shuffle:
        random_order = np.arange(inputs_size)
        np.random.shuffle(random_order)
        inputs, targets = inputs[random_order, :], targets[random_order]
    num_blocks = int(inputs_size / batch_size)
    for i in range(num_blocks):
        yield inputs[i * batch_size: (i+1) * batch_size, :], targets[i * batch_size: (i+1) * batch_size]
    if num_blocks * batch_size != inputs_size:
        yield inputs[num_blocks * batch_size:, :], targets[num_blocks * batch_size:]


def multi_data_loader(inputs, targets, batch_size, shuffle=True):
    """
    Both inputs and targets are list of numpy arrays, containing instances and labels from multiple sources.
    """
    assert len(inputs) == len(targets)
    input_sizes = [data.shape[0] for data in inputs]
    max_input_size = max(input_sizes)
    num_domains = len(inputs)
    if shuffle:
        for i in range(num_domains):
            r_order = np.arange(input_sizes[i])
            np.random.shuffle(r_order)
            inputs[i], targets[i] = inputs[i][r_order], targets[i][r_order]
    num_blocks = int(max_input_size / batch_size)
    for j in range(num_blocks):
        xs, ys = [], []
        for i in range(num_domains):
            ridx = np.random.choice(input_sizes[i], batch_size)
            xs.append(inputs[i][ridx])
            ys.append(targets[i][ridx])
        yield xs, ys

