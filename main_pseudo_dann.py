import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# from model import DarnMLP
from model_pseudo_dann import DarnMLP
from tqdm import tqdm
from load_data import load_numpy_data, data_loader, multi_data_loader
import utils
from anndata import AnnData
import umap
import scanpy as sc
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./log")

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Name of the dataset.",
                    type=str, choices=['pbmc_ding', 'pancreas'], default="pbmc_ding")
parser.add_argument("--result_path", help="Where to save results.",
                    type=str, default="./results")
parser.add_argument("--lr", help="Learning rate.",
                    type=float, default=0.5)
parser.add_argument("--mu", help="Hyperparameter of the coefficient for the domain adversarial loss.",
                    type=float, default=1e-3)
parser.add_argument("--gamma", help="Inverse temperature hyperparameter.",
                    type=float, default=1.0)
parser.add_argument("--epoch", help="Number of training epochs.",
                    type=int, default=50)
parser.add_argument("--batch_size", help="Batch size during training.",
                    type=int, default=64)
parser.add_argument("--cuda", help="Which cuda device to use.",
                    type=int, default=2)
parser.add_argument("--seed", help="Random seed.",
                    type=int, default=666)
parser.add_argument("--mode", help="Aggregation mode [dynamic|L2]: L2 for DARN, dynamic for MDAN.",
                    type=str, choices=['dynamic', 'L2'], default="L2")
parser.add_argument("--dim", help="hidden dim.",
                    type=str, default="1024-512-32")
parser.add_argument("--filtered", help="filter pseudo label.",
                    type=str, choices=['T', 'F'], default="F")
args = parser.parse_args()
args.filtered=args.filtered=='T'
args.dim=list(map(lambda i:int(i),args.dim.split('-')))
device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size

result_path = os.path.join(args.result_path,
                           args.name)
if not os.path.exists(result_path):
    os.makedirs(result_path)

logger = utils.get_logger(os.path.join(result_path,
                                       "gamma_%g_seed_%d_batch_%d_lr_%g.log" % (args.gamma,
                                                                 args.seed,args.batch_size,args.lr)))
logger.info("Hyperparameter setting = %s" % args)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

#################### Loading the datasets ####################

time_start = time.time()


def scanpy_umap(embeddings, labels, batches, name="default"):
    formatting = AnnData(embeddings)
    formatting.obs["cell_type"] = [str(x) for x in labels]
    formatting.obs["batches"] = [str(x) for x in batches]

    fontsize = 10
    sc.pp.neighbors(formatting, n_neighbors=25, use_rep='X', n_pcs=40)
    sc.tl.umap(formatting)
    ax=sc.pl.umap(
        formatting, color="cell_type",
        legend_fontsize=fontsize,show=False,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(None)
    ax.get_figure().savefig(f'./figures/celltype_{name}.png',dpi=200,bbox_inches='tight')
    
    ax=sc.pl.umap(
        formatting, color="batches",
        legend_fontsize=fontsize,show=False,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(None)
    ax.get_figure().savefig(f'./figures/batches_{name}.png',dpi=200,bbox_inches='tight')
    
def umap_combined_data(num_src_domains, model, source_insts, source_labels, final_test_data, final_test_labels, train_data_names, rename, name):
    model.eval()

    combined_embedding = None
    combined_label = []
    combined_batch = []
    batch_rename={
        "pbmcsca_10x_Chromium_v2":"10x_v2",
        "pbmcsca_10x_Chromium_v3":"10x_v3",
        "pbmcsca_CEL-Seq2":"CEL-Seq2",
        "pbmcsca_Seq-Well":"Seq-Well",
        "pbmcsca_Smart-seq2":"Smart-seq2",
        "pbmcsca_Drop-seq":"Drop-seq",
        "pbmcsca_inDrops":"inDrops",
    }
    for i in range(num_src_domains):
        if combined_embedding is None:
            combined_embedding = model.get_embedding( torch.tensor(source_insts[i], requires_grad=False).to(device)).cpu().detach().numpy()
        else:
            combined_embedding = np.concatenate((combined_embedding,model.get_embedding( torch.tensor(source_insts[i], requires_grad=False).to(device)).cpu().detach().numpy()),
                                                axis=0)
        combined_label.extend([rename[j] for j in source_labels[i]])
        combined_batch.extend([batch_rename[train_data_names[i]]] * len(source_labels[i]))

    combined_embedding=np.concatenate((combined_embedding, model.get_embedding( torch.tensor(final_test_data, requires_grad=False).to(device)).cpu().detach().numpy()), axis=0)
    combined_label.extend([rename[j] for j in final_test_labels])
    combined_batch.extend([batch_rename[name]] * len(list(final_test_labels)))

    scanpy_umap(combined_embedding, combined_label, combined_batch, name)



def test_step_result(model, final_test_data, final_test_label, writer, data_name, epoch, name):
    model.eval()

    test_loader = data_loader(final_test_data, final_test_label, batch_size=1000, shuffle=False)

    test_acc = 0.
    y_true = np.array([])
    y_pred = np.array([])
    for xt, yt in test_loader:
        xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
        yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
        preds_labels = torch.squeeze(torch.max(model.inference(xt), 1)[1])
        test_acc += torch.sum(preds_labels == yt).item()

        yt = yt.cpu().numpy()
        preds_labels = preds_labels.cpu().numpy()

        y_true = np.concatenate((y_true, yt))
        y_pred = np.concatenate((y_pred, preds_labels))

    test_acc /= final_test_data.shape[0]
    logger.info("Test accuracy on %s = %.6g" % (data_name, test_acc))
    macro = f1_score(y_true, y_pred, average='macro')
    logger.info("Test macro-F1 on %s = %.6g\n" % (data_name, macro))


    writer.add_scalar('Test/' + name + '/acc/' + data_name, test_acc, epoch)
    writer.flush()
    writer.add_scalar('Test/' + name + '/macro/' + data_name, macro, epoch)
    writer.flush()


def test_result(model, final_test_data, final_test_label, data_name):
    model.eval()

    test_loader = data_loader(final_test_data, final_test_label, batch_size=1000, shuffle=False)
    test_acc = 0.
    y_true = np.array([])
    y_pred = np.array([])
    for xt, yt in test_loader:
        xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
        yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
        preds_labels = torch.squeeze(torch.max(model.inference(xt), 1)[1])
        test_acc += torch.sum(preds_labels == yt).item()

        yt = yt.cpu().numpy()
        preds_labels = preds_labels.cpu().numpy()

        y_true = np.concatenate((y_true, yt))
        y_pred = np.concatenate((y_pred, preds_labels))

    test_acc /= final_test_data.shape[0]
    logger.info("Test accuracy on %s = %.6g" % (data_name, test_acc))
    macro = f1_score(y_true, y_pred, average='macro')
    logger.info("Test macro-F1 on %s = %.6g\n" % (data_name, macro))


data_names, train_insts, train_labels, _, _, configs,rename = load_numpy_data(
    args.name,
    logger,
)
configs["hidden_layers"]=args.dim
configs["mode"] = args.mode
configs['mu'] = args.mu
configs["gamma"] = args.gamma
configs["num_src_domains"] = len(data_names) - 1
num_datasets = len(data_names)

logger.info("Time used to process the %s = %g seconds." % (args.name, time.time() - time_start))
logger.info("-" * 100)

test_results = {}
np_test_results = np.zeros(num_datasets)


num_src_domains = configs["num_src_domains"]
logger.info("Model setting = %s." % configs)

alpha_list = np.zeros([num_datasets, num_src_domains, args.epoch])
avg=[]
for i in range(num_datasets):

    # Build source instances
    source_insts = []
    source_labels = []
    train_data_names = []
    for j in range(num_datasets):
        if j != i:
            source_insts.append(train_insts[j].astype(np.float32))
            source_labels.append(train_labels[j].astype(np.int64))

            train_data_names.append(data_names[j])

    # Build target instances
    target_insts = train_insts[i].astype(np.float32)
    target_labels = train_labels[i].astype(np.int64)

    final_test_data = target_insts
    final_test_label = target_labels

    # Model
    model = DarnMLP(configs).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Training phase
    model.train()
    time_start = time.time()
    for t in range(args.epoch):

        running_loss = 0.0
        train_loader = multi_data_loader(source_insts, source_labels, batch_size)

        for xs, ys in train_loader:

            for j in range(num_src_domains):

                xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)

            ridx = np.random.choice(target_insts.shape[0], batch_size)
            tinputs = target_insts[ridx, :]
            tinputs = torch.tensor(tinputs, requires_grad=False).to(device)

            tinputs_labels = target_labels[ridx]
            tinputs_labels = torch.tensor(tinputs_labels, requires_grad=False).to(device)

            optimizer.zero_grad()
            loss, alpha = model(xs, ys, tinputs, tinputs_labels, mode = 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logger.info("Epoch %d, Alpha on %s: %s" % (t, data_names[i], alpha))
        alpha_list[i, :, t] = alpha

        logger.info("Epoch %d, loss = %.6g" % (t, running_loss))
        writer.add_scalar('Train/Loss_'+data_names[i], running_loss, t)
        writer.flush()
        test_step_result(model, final_test_data, final_test_label, writer, data_names[i], t, 'normal')

        if t > 19:
            if t % 10 == 0:
                target = torch.tensor(target_insts).to(device)
                prob=model.out(target)
                if args.filtered:
                    idx=prob.max(1)[0].argsort()
                    idx=idx[:int(0.7*len(idx))]
                    target_presudo_labels = prob[idx].argmax(1)
                    tgt=target_insts[idx.cpu()]
                else:
                    target_presudo_labels = prob.argmax(1)
                    tgt=target_insts[:]
                    
                presudo_loader = data_loader(tgt, target_presudo_labels, batch_size=batch_size, shuffle=True)
            
            running_loss = 0.0

            for xs, ys in presudo_loader:
                xs = torch.tensor(xs, requires_grad=False).to(device)
                ys = torch.tensor(ys, requires_grad=False).to(device)

                optimizer.zero_grad()
                loss = model(None, None, tinputs=xs, tipout=ys, mode = 2)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            writer.add_scalar('Pseudo/Loss_'+data_names[i], running_loss, t)
            writer.flush()
    
    logger.info("Finish training %s in %.6g seconds" % (data_names[i],
                                                        time.time() - time_start))
    model.eval()

    test_loader = data_loader(final_test_data, final_test_label, batch_size=1000, shuffle=False)

    test_acc = 0.
    y_true = np.array([])
    y_pred = np.array([])

    for xt, yt in test_loader:
        xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
        yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
        preds_labels = torch.squeeze(torch.max(model.inference(xt), 1)[1])
        test_acc += torch.sum(preds_labels == yt).item()

        yt = yt.cpu().numpy()
        preds_labels = preds_labels.cpu().numpy()

        y_true = np.concatenate((y_true, yt))
        y_pred = np.concatenate((y_pred, preds_labels))


    test_acc /= final_test_data.shape[0]
    logger.info("Test accuracy on %s = %.6g" % (data_names[i], test_acc))
    test_results[data_names[i]] = test_acc
    np_test_results[i] = test_acc
    rn={j:i for i,j in rename.items()}
    df=pd.DataFrame({'pred':y_pred,'true':y_true}).replace(rn)
    df.to_csv(f'./output/{data_names[i]}-output.csv')
    acc = accuracy_score(y_true, y_pred)
    print("acc: ", acc)
    avg.append(acc)
    macro_list = f1_score(y_true, y_pred, average=None)
    print("macro list: ", macro_list)

    macro = f1_score(y_true, y_pred, average='macro')
    print("F1 macro: ", macro)

    for j in range(num_src_domains):
        test_result(model, source_insts[j], source_labels[j], data_names[j+1])

    umap_combined_data(num_src_domains, model, source_insts, source_labels, final_test_data, final_test_label, train_data_names, rn, f'{data_names[i]}')
logger.info("All test accuracies: ")
logger.info(test_results)
logger.info("Average test accuracies: ")
logger.info(sum(avg)/len(avg))
# Save results to files
test_file = os.path.join(result_path,
                         "gamma_%g_seed_%d_test.txt" % (args.gamma,
                                                        args.seed))
np.savetxt(test_file, np_test_results, fmt='%.6g')


logger.info("Done")
logger.info("*" * 100)
