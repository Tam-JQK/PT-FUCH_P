import torch
import numpy as np
import h5py
import scipy.io as scio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from models import CNNFImgModule
import copy
from fast_pytorch_kmeans import KMeans
from torch.autograd import Variable
import torch.nn.functional as F
import math
import random

def average_weights(w):

    w_avg = copy.deepcopy(w)

    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.div(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

def agg_func(protos):
    """
    Average the protos for each local user
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = torch.stack(proto_list)
        agg_protos[label] = torch.mean(proto, dim=0).data

    return agg_protos

def global_proto_cluster(local_protos_dict):

    protos_bank = []
    for k,v in local_protos_dict.items():
        if bool(torch.isnan(v[0]).any()):
            continue
        else:
            protos_bank.append(v[0])

    protos_bank = torch.cat(protos_bank, dim=0).contiguous()
    protos_bank = F.normalize(protos_bank,dim=1)

    cluster_result = {'inst2cluster': [], 'centroids': []}
    kmeans = KMeans(n_clusters=24,mode='cosine')#, verbose=1)
    cluster_r = kmeans.fit_predict(protos_bank)
    cc = kmeans.centroids
    cluster_result['inst2cluster'].append(cluster_r)
    cluster_result['centroids'].append(cc)
    return cluster_result['centroids']

def compress(train_loader, test_loader, model_I, model_T,device):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(train_loader):
        with torch.no_grad():
            var_data_I = Variable(data_I.to(device))
            _, code_I = model_I(var_data_I)
            # code_I = fc1(code_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).to(device))
        _, code_T = model_T(var_data_T)
        # code_T = fc2(code_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())
    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, data_L, _) in enumerate(test_loader):
        with torch.no_grad():
            var_data_I = Variable(data_I.to(device))
            _, code_I = model_I(var_data_I)
            # code_I = fc1(code_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).to(device))
        _, code_T = model_T(var_data_T)
        # code_T = fc2(code_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())
    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.array(qu_L)
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):

    num_query = qu_L.shape[0]
    topkmap = 0
    maps = []
    ids = []
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
        maps.append(topkmap_)
        ids.append(iter)
    topkmap = topkmap / num_query
    return topkmap

def calculate_hamming(B1, B2):

    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def load_pretrain_model(path):
    return scio.loadmat(path)

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        
        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count

def split_data(database_s,images,tags,labels):
    query_size = 2000
    training_size = 5000
    database_size = database_s
    
    X = {}
    X['query'] = images[0: query_size]
    X['train'] = images[query_size: training_size + query_size]
    X['retrieval'] = images[query_size: query_size + database_size]

    Y = {}
    Y['query'] = tags[0: query_size]
    Y['train'] = tags[query_size: training_size + query_size]
    Y['retrieval'] = tags[query_size: query_size + database_size]

    L = {}
    L['query'] = labels[0: query_size]
    L['train'] = labels[query_size: training_size + query_size]
    L['retrieval'] = labels[query_size: query_size + database_size]

    return X,Y,L


def load_data(args,path):
    file = h5py.File(path)
    images = file['IAll'][:]
    labels = file['LAll'][:]
    tags = file['TAll'][:]
    images = images.transpose(3,2,0,1)
    labels = labels.transpose(1,0)
    tags = tags.transpose(1,0)

    file.close()

    pretrain_model = load_pretrain_model("./imagenet-vgg-f.mat")
   
    FeatNet_I = CNNFImgModule(pretrain_model)
    FeatNet_I.cuda().eval()
    num_data = len(images)
    new_images = np.zeros((num_data, 4096))
    for i in range(num_data):
        feature = FeatNet_I(torch.Tensor(images[i]).unsqueeze(0).cuda())
        new_images[i] = feature.cpu().detach().numpy()
    images = new_images.astype(np.float32)

    tags = tags.astype(np.float32)

    labels = labels.astype(int)


    


    return images,tags,labels

def getdataset(args,database_s,data_path):
    images, tags, labels = load_data(args,data_path)

    X, Y, L = split_data(database_s,images, tags, labels)
    print('...loading and splitting data finish')

    train_L = L['train']
    train_x = X['train']
    train_y = Y['train']

    query_L = L['query']
    query_x = X['query']
    query_y = Y['query']

    retrieval_L = L['retrieval']
    retrieval_x = X['retrieval']
    retrieval_y = Y['retrieval']

    


    return  train_L,train_x,train_y,retrieval_L,retrieval_x,retrieval_y,query_L,query_x,query_y

def avg_divide(l, g):

    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist

def split_list_by_idcs(l, idcs):

    res = []
    current_index = 0
    for index in idcs: 
        res.append(l[current_index: index])
        current_index = index

    return res

def mixture_distribution_split_noniid(dataset, n_classes, n_clients, n_clusters, alpha):
    rng = np.random.RandomState(0)

    if n_clusters == -1:
        n_clusters = n_classes

    all_labels = list(range(n_classes))

    np.random.shuffle(all_labels)

    clusters_labels = avg_divide(all_labels, n_clusters)


    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    
    data_idcs = list(range(len(dataset)))
    
    #

    clusters_sizes = np.zeros(n_clusters, dtype=int)

    clusters = {k: [] for k in range(n_clusters)}
    

    for idx in data_idcs:
        A,B,label,D = dataset[idx]

        labelss = []
        for j in range(len(label.tolist())):
            if label[j] == 1:
                labelss.append(j)
      

        labels = np.array(labelss)
        for la in labels:
            group_id = label2cluster[la]

            clusters_sizes[group_id] += 1

            clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64) 

   
    for cluster_id in range(n_clusters):
       
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
    
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)


    
    clients_idcs = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
     
        cluster_split = split_list_by_idcs(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, idcs in enumerate(cluster_split):
            clients_idcs[client_id] += idcs
    
    return clients_idcs


def train_dirichlet_split_noniid(train_labels, alpha, n_clients):
   
    labelss = []
    for i in range(len(train_labels)):
        for j in range(len(train_labels[i])):
            if train_labels[i][j] == 1:
                labelss.append(j)

    labels = np.array(labelss)

    n_classes = n_clients

    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)


    class_idcs = [np.argwhere(labels==y).flatten() 
           for y in range(n_classes)]


    

    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def one_hot(x,class_count):
    return torch.eye(class_count)[x,:]

def get_embedding_Kmeans(corpus):
    corpus_embeddings = []

    corpus_embeddings = corpus

    num_clusters = 10
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_


    return cluster_assignment

def get_persolabels(number_of_clients,images):
    cluster_assignment = get_embedding_Kmeans(images)
    class_count = number_of_clients
    labels_cluster = one_hot(cluster_assignment.astype('int64'),class_count=class_count)

    
    return labels_cluster

def prepare_data_noniid(args):
    datapath = "/flickr.mat"
    database_s = 18016
    train_L,train_x,train_y,retrieval_L,retrieval_x,retrieval_y,query_L,query_x,query_y= getdataset(args,database_s,datapath)
    n_classes = 24

    n_clients = args.num_users
    n_clusters = args.num_users
    alpha = 0.1

   
    
    dataset_train = [train_L,train_x,train_y]
    dataset_test = [retrieval_L,retrieval_x,retrieval_y,query_L,query_x,query_y]


    train_cluster_label = get_persolabels(n_clients,train_x)
    client_idcs_train = train_dirichlet_split_noniid(train_cluster_label, alpha=alpha, n_clients=n_clients)




    retireval_cluster_label = get_persolabels(n_clients,retrieval_x)
    client_idcs_retrieval = train_dirichlet_split_noniid(retireval_cluster_label, alpha=alpha, n_clients=n_clients)




    query_cluster_label = get_persolabels(n_clients,query_x)
    client_idcs_qery = train_dirichlet_split_noniid(query_cluster_label, alpha=alpha, n_clients=n_clients)


    return dataset_train,dataset_test, client_idcs_train,client_idcs_retrieval,client_idcs_qery

def update_queue(queue,fuse,use_the_queue = True):
    bs = 256
    fuse2 = fuse.detach()
    out = fuse.detach()
    if queue is not None:  # no queue in first round
        if use_the_queue or not torch.all(queue[ -1, :] == 0):  # queue[2,3840,128] if never use the queue or the queue is not full
            use_the_queue = True
            # print('use queue')
            out = torch.cat((queue,fuse.detach()))  # queue [1920*128] w_t [128*3000] = 1920*3000 out [32*3000] 1952*3000

        if  fuse.shape[0] == bs:  
            queue[bs:] = queue[ :-bs].clone()  # move 0-6 to 1-7 place
            queue[:bs] = fuse2
        else:
            queue = out
    return queue,out

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def Calculate_mAP(tst_label, tst_binary, db_label, db_binary, top_k=-1):
    if top_k == -1:
        top_k = db_binary.size(0)
    mAP = 0.

    num_query = tst_binary.size(0)
    NS = (torch.arange(top_k) + 1).float()
    for idx in range(num_query):
        query_label = tst_label[idx].unsqueeze(0)
        query_binary = tst_binary[idx]
        result = query_label.mm(db_label.t()).squeeze(0) > 0

        hamm = calc_hammingDist(query_binary, db_binary)
        _, index = hamm.sort()
        index.squeeze_()
        result = result[index[: top_k]].float()
        tsum = torch.sum(result)
        if tsum == 0:
            continue

        all_similar = torch.sum(result)

        accuracy = torch.cumsum(result, dim=0) / torch.sum(result)
        mAP += float(torch.sum(result * accuracy / NS).item())
    return mAP/num_query

def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]).cuda())
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]))
    return lt

def generate_hash_code(dataloader, img_net, txt_net):
    bs, tags, clses = [], [], []
    for img, tag, label, _ in dataloader:
        clses.append(label)
        # if use_gpu:
        img, tag = img.cuda(), tag.cuda()
        img = img.type(torch.float)
        img = img_net(img).cpu().data
        bs.append(img)

        tag = tag.unsqueeze(1).unsqueeze(-1).type(torch.float)
        tag = txt_net(tag).cpu().data
        tags.append(tag)
    return torch.sign(torch.cat(bs)), torch.sign(torch.cat(tags)), torch.cat(clses)


def calculate_map(qu_B, re_B, qu_L, re_L):

    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum =np.int(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map