import argparse
import random
import numpy as np
import torch
from torch import nn
from utils import *
from NCECriterion import NCESoftmaxLoss
from NCEAverage import NCEAverage
from losses import ContrastiveLoss
from fast_pytorch_kmeans import KMeans
import torch.nn.functional as F
from ImageNet import ImageNet
from TextNet import TextNet
from torch.nn.utils.clip_grad import clip_grad_norm_
import os.path as osp
import math



def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--temperature", type=float, default=0.26, help="the temperature parameter for contrastive loss")
    parser.add_argument("--ts", type=float, default=0.81, help="the temperature parameter for js loss in teacher model")
    parser.add_argument("--ss", type=float, default=0.81, help="the temperature parameter for js loss in student model")

    parser.add_argument('--rounds', type=int, default=100, help="number of rounds of training")
    parser.add_argument('--train_ep', type=int, default=10, help="the number of local episodes: E")
    parser.add_argument('--train_ep_private', type=int, default=10, help="the number of local episodes: E")
    parser.add_argument('--learning_rate_img_private', type=float, default=1e-5, metavar='N',help='learning_rate')#0.01
    parser.add_argument('--learning_rate_txt_private', type=float, default=1e-5, metavar='N',help='learning_rate')
    parser.add_argument('--weight_decay_p', default=1e-6, type=float, help='weight_decay')
    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    parser.add_argument('--bit', type=int, default=16, help="hash code length")
    parser.add_argument('--gpu', default=2, type=int, help="index of gpu")

    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--alg', type=str, default='FedUCCH', help="algorithms")
    parser.add_argument('--noniid', type=str, default='--', help="noniid")
    parser.add_argument('--batch_size', type=int, default=256, help="local batch size")
    parser.add_argument('--learning_rate_img', type=float, default=1e-4, metavar='N',help='learning_rate')#0.01
    parser.add_argument('--learning_rate_txt', type=float, default=1e-4, metavar='N',help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.4, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight_decay')
    parser.add_argument('--device', default="cuda", type=str, help="cpu, cuda, or others")
    parser.add_argument('--optimizer', type=str, default='Adam', help="type of optimizer")
    parser.add_argument('--save_model_path', default='./checkpoint/', help='path to save_model')
    parser.add_argument('--nce', type=int, default=0, help="nce")
    parser.add_argument('--dataset', type=str, default='MIR-FLICKR25K', help="name of dataset, e.g. MIR-FLICKR25K")
    parser.add_argument('--data_dir', type=str, default='./data/', help="name of dataset, default: './data/'")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--feature_iid', type=int, default=0, help='Default set to feature non-IID. Set to 1 for feature IID.')
    parser.add_argument('--label_iid', type=int, default=0, help='Default set to label non-IID. Set to 1 for label IID.')
    parser.add_argument('--alpha', type=float, default=.9)
    parser.add_argument('--K', type=int, default=4096)
    parser.add_argument('--T', type=float, default=.9)
    parser.add_argument('--shift', type=float, default=1)
    parser.add_argument('--margin', type=float, default=.2)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--num_hiden_layers', default=[3, 2], nargs='+', help='<Required> Number of hiden lyaers')
    args = parser.parse_args()
    return args

class LocalUpdateDJSRH(object):
    def __init__(self, args, id,dataset,idxs):
        self.id = id
        self.args = args
        self.dataset = dataset#[train_L,train_x,train_y]
        self.m_size = len(idxs)
        dataloader_train = self.train_val_test(dataset,idxs)
        self.dataloader_train = dataloader_train['train']
        self.device = args.device
        self.loss_mse = nn.MSELoss().to(self.device)

        #UCCH
        self.criterion = ContrastiveLoss(args.margin, shift=args.shift)
        self.contrast = NCEAverage(args.bit, self.m_size, args.K, args.T, args.momentum)
        criterion_contrast = NCESoftmaxLoss()
        self.contrast = self.contrast.cuda()
        self.criterion_contrast = criterion_contrast.cuda()
 
        e_size = self.args.bit
        queue_l = self.args.batch_size
        self.queue_v_global = torch.zeros(
                        queue_l,
                        e_size,
                    ).cuda()
        self.queue_v_local = torch.zeros(
                queue_l,
                e_size,
            ).cuda()
        
    def train_val_test(self, dataset, train_idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        
        train_L =dataset[0]
        train_x = dataset[1]
        train_y = dataset[2]

        train_X_list = []
        train_Y_list = []
        train_L_list = []
        for idx in train_idxs:
            # print(type(idx))
            train_X_list.append(train_x[idx])
            train_Y_list.append(train_y[idx])
            train_L_list.append(train_L[idx])
        train_X = np.array(train_X_list)
        train_Y = np.array(train_Y_list)
        train_L = np.array(train_L_list)

        imgs = {'train': train_X}
        texts = {'train': train_Y}
        labels = {'train': train_L}

        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                for x in ['train']}

        shuffle = {'train': True}

        dataloader_train = {x: DataLoader(dataset[x], batch_size=self.args.batch_size,
                                    shuffle=shuffle[x], num_workers=4) for x in ['train']}

        return dataloader_train
    
    def update_weights_UCCH_new(self,args,idx,global_protos,model_img,model_txt,private_img_model,private_txt_model):
   
        global_img_model = model_img
        global_txt_model = model_txt
        
        device = self.device

        global_img_model.to(device)
        global_txt_model.to(device)
        
        private_img_model.to(device)
        private_txt_model.to(device)

        #cross-modal relation KD loss
        src_model_I = copy.deepcopy(model_img)
        src_model_T = copy.deepcopy(model_txt)
        src_model_I.to(device)
        src_model_T.to(device)
        src_model_I.freeze_grad()
        src_model_T.freeze_grad()


        if args.optimizer == "sgd":
            optimizer_i_s = torch.optim.SGD(filter(lambda p: p.requires_grad, model_img.parameters()), lr=args.learning_rate_img)
            optimizer_t_s = torch.optim.SGD(filter(lambda p: p.requires_grad, model_txt.parameters()), lr=args.learning_rate_txt)
            optimizer_i_private = torch.optim.SGD(filter(lambda p: p.requires_grad, private_img_model.parameters()), lr=args.learning_rate_img_private)
            optimizer_t_private = torch.optim.SGD(filter(lambda p: p.requires_grad, private_txt_model.parameters()), lr=args.learning_rate_txt_private)
        else:
            optimizer_i = torch.optim.Adam(global_img_model.parameters(), lr=args.learning_rate_img, weight_decay=args.weight_decay)
            optimizer_t = torch.optim.Adam(global_txt_model.parameters(), lr=args.learning_rate_txt, weight_decay=args.weight_decay)
            optimizer_i_p = torch.optim.Adam(private_img_model.parameters(), lr=args.learning_rate_img, weight_decay=args.weight_decay)
            optimizer_t_p = torch.optim.Adam(private_txt_model.parameters(), lr=args.learning_rate_txt, weight_decay=args.weight_decay)

        private_img_model.train()
        private_txt_model.train()
        for epoch in range(args.train_ep_private):
            for batch_idx, (img_F,txt_F,labels,index) in enumerate(self.dataloader_train):
                img = Variable(img_F.to(device))
                txt = Variable(torch.FloatTensor(txt_F.numpy()).to(device))
                idx = index.to(device)
                batch_size = img.size(0)
                hid_I_p,images_outputs = private_img_model(img)
                hid_T_p,texts_outputs = private_txt_model(txt)

                out_l, out_ab = self.contrast(images_outputs, texts_outputs, idx, epoch=epoch-args.warmup_epoch)
                l_loss = self.criterion_contrast(out_l)
                ab_loss = self.criterion_contrast(out_ab)
                Lc = l_loss + ab_loss
                Lr = self.criterion(images_outputs, texts_outputs)
                loss = Lc * args.alpha + Lr * (1. - args.alpha)

                fushed = (hid_I_p + hid_T_p) / 2
                fushed = F.normalize(fushed)

                queue_v,out = update_queue(self.queue_v_local,fushed)

                kmeans = KMeans(n_clusters=24,mode='cosine')#, verbose=1)
                peso_labels = kmeans.fit_predict(queue_v)
                centroid = kmeans.centroids

                loss_cluster = self.cluster_contrast(hid_I_p,centroid.to(device),peso_labels[-batch_size:].to(device),batch_size)+\
                                    self.cluster_contrast(hid_T_p,centroid.to(device),peso_labels[-batch_size:].to(device),batch_size)

                _, code_I_tea = src_model_I(img)
                _, code_T_tea = src_model_T(txt)

                img_stu_norm = F.normalize(images_outputs)
                txt_stu_norm = F.normalize(texts_outputs)
                img_tea_norm = F.normalize(code_I_tea)
                txt_tea_norm = F.normalize(code_T_tea)

                loss_kd_s = self.loss_mse(img_stu_norm.mm(img_stu_norm.t()), img_tea_norm.mm(img_tea_norm.t())) + \
                            self.loss_mse(txt_stu_norm.mm(txt_stu_norm.t()), txt_tea_norm.mm(txt_tea_norm.t())) + \
                            self.loss_mse(img_stu_norm.mm(txt_stu_norm.t()), img_tea_norm.mm(txt_tea_norm.t())) + \
                            self.loss_mse(txt_stu_norm.mm(img_stu_norm.t()), txt_tea_norm.mm(img_tea_norm.t()))
                
                loss_kd = loss_kd_s / loss_cluster

                loss_r = loss_cluster + loss_kd
                loss = loss +loss_r
                optimizer_i_p.zero_grad()
                optimizer_t_p.zero_grad()
                loss.backward()
                clip_grad_norm_(private_img_model.parameters(), 1.)
                clip_grad_norm_(private_txt_model.parameters(), 1.)
                optimizer_i_p.step()
                optimizer_t_p.step()

        global_img_model.train()
        global_txt_model.train()
        private_img_model.eval()
        private_txt_model.eval()

        for epoch in range(args.train_ep):
            batch_loss = []
            for batch_idx, (img_F,txt_F,labels,index) in enumerate(self.dataloader_train):
                img = Variable(img_F.to(device))
                txt = Variable(torch.FloatTensor(txt_F.numpy()).to(device))
                idx = index.to(device)

                hid_I,images_outputs_g = global_img_model(img)
                hid_T,texts_outputs_g = global_txt_model(txt)
   
                out_l, out_ab = self.contrast(images_outputs_g, texts_outputs_g, idx, epoch=epoch-args.warmup_epoch)
                l_loss = self.criterion_contrast(out_l)
                ab_loss = self.criterion_contrast(out_ab)
                Lc = l_loss + ab_loss
                Lr = self.criterion(images_outputs_g, texts_outputs_g)
                loss = Lc * args.alpha + Lr * (1. - args.alpha)

                loss_RG = 0*loss
                loss_RL = 0*loss
                if len(global_protos) != 0:
                    hid_I_p, code_I_p = private_img_model(img)
                    hid_T_p, code_T_p = private_txt_model(txt)
                    fushed = (hid_I_p + hid_T_p) / 2
                    fushed = F.normalize(fushed)
                    queue_v,out = update_queue(self.queue_v_local,fushed)
                    kmeans = KMeans(n_clusters=24,mode='cosine')#, verbose=1)
                    cluster_r = kmeans.fit_predict(queue_v)
                    local_protos = kmeans.centroids
              
                    loss_RG=self.js_loss(hid_I,hid_T,global_protos[0],t=self.args.temperature, t2=self.args.ts)
                    loss_RL=self.js_loss(hid_I,hid_T,local_protos,t=self.args.temperature, t2=self.args.ss)
                       
                loss = loss  + loss_RL +  loss_RG
                optimizer_i.zero_grad()
                optimizer_t.zero_grad()
                loss.backward()
                clip_grad_norm_(global_img_model.parameters(), 1.)
           
                optimizer_i.step()
                optimizer_t.step()

        global_img_model.eval()
        global_txt_model.train()
        for epoch in range(args.train_ep):
            batch_loss = []
            for batch_idx, (img_F,txt_F,labels,index) in enumerate(self.dataloader_train):
                img = Variable(img_F.to(device))
                txt = Variable(torch.FloatTensor(txt_F.numpy()).to(device))
                idx = index.to(device)

                hid_I,images_outputs_g = global_img_model(img)
                hid_T,texts_outputs_g = global_txt_model(txt)
     
                out_l, out_ab = self.contrast(images_outputs_g, texts_outputs_g, idx, epoch=epoch-args.warmup_epoch)
                l_loss = self.criterion_contrast(out_l)
                ab_loss = self.criterion_contrast(out_ab)
                Lc = l_loss + ab_loss
                Lr = self.criterion(images_outputs_g, texts_outputs_g)
                loss = Lc * args.alpha + Lr * (1. - args.alpha)
                optimizer_t.zero_grad()
                loss.backward()
                optimizer_t.step()

              
        with torch.no_grad():
            private_img_model.eval()
            private_txt_model.eval()

            proj_bank_fuse = []
            n_samples = 0
            for bs,(images,texts,_,indd) in enumerate(self.dataloader_train):
                img = Variable(images.to(device))
                txt = Variable(torch.FloatTensor(texts.numpy()).to(device))
                if(n_samples >= self.m_size):
                    break
                hid_I,_ = private_img_model(img)
                hid_T,_ = private_txt_model(txt)
                H = (hid_I + hid_T)/2
                H = F.normalize(H)
                proj_bank_fuse.append(H)
                n_samples += len(indd)
            proj_bank_img = torch.cat(proj_bank_fuse, dim=0).contiguous()
            if(n_samples > self.m_size):
                proj_bank_img = proj_bank_img[:self.m_size]
            cluster_result = {'inst2cluster': [], 'centroids': []}
            kmeans = KMeans(n_clusters=24,mode='cosine')#, verbose=1)
            cluster_r = kmeans.fit_predict(proj_bank_img)
            cc = kmeans.centroids
            cluster_result['inst2cluster'].append(cluster_r)
            cluster_result['centroids'].append(cc)
        
        return global_img_model.cpu().state_dict(), global_txt_model.cpu().state_dict(),private_img_model.cpu().state_dict(), private_txt_model.cpu().state_dict(),cluster_result['centroids']

    def cluster_contrast(self,fushed,centroid,labels,bs):
      
      S = torch.matmul(fushed, centroid.t())

      target = torch.zeros(bs,centroid.shape[0]).to(S.device)

      target[range(target.shape[0]), labels] = 1

      S = S - target * (0.001)

      if self.args.nce==0:
          I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), labels)

      else:
          S = S.view(S.shape[0], S.shape[1], -1)
          nominator = S * target[:, :, None]
          nominator = nominator.sum(dim=1)
          nominator = torch.logsumexp(nominator, dim=1)
          denominator = S.view(S.shape[0], -1)
          denominator = torch.logsumexp(denominator, dim=1)
          I2C_loss = torch.mean(denominator - nominator)

      return I2C_loss
  
    def js_loss(self,x1, x2, xa, t=0.1, t2=0.01):

        pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
        inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
        pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
        inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
        target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2
        js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
        js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
        return (js_loss1 + js_loss2) / 2.0

class LocalTest(object):
    def __init__(self, args,id,dataloaders,idcs_retrieval,idcs_query):
        self.args = args

        self.testloader,self.databaseloader = self.test_split(args,dataloaders,idcs_retrieval[id],idcs_query[id])
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, args,dataloader,idcs_retrieval,idcs_query):

        retrieval_L = dataloader[0]
        retrieval_x = dataloader[1]
        retrieval_y =dataloader[2]
        
        query_L = dataloader[3]
        query_x = dataloader[4]
        query_y = dataloader[5]

        retrieval_X_list = []
        retrieval_Y_list = []
        retrieval_L_list = []
        for idx in idcs_retrieval:
    
            retrieval_X_list.append(retrieval_x[idx])
            retrieval_Y_list.append(retrieval_y[idx])
            retrieval_L_list.append(retrieval_L[idx])

        retrieval_x_for_this_client = np.array(retrieval_X_list)
        retrieval_y_for_this_client = np.array(retrieval_Y_list)
        retrieval_L_for_this_client = np.array(retrieval_L_list)
        
        query_X_list = []
        query_Y_list = []
        query_L_list = []
        for idx in idcs_query:
         
            query_X_list.append(query_x[idx])
            query_Y_list.append(query_y[idx])
            query_L_list.append(query_L[idx])

        query_x_for_this_client = np.array(query_X_list)
        query_y_for_this_client = np.array(query_Y_list)
        query_L_for_this_client = np.array(query_L_list)


        imgs = {'query': query_x_for_this_client, 'database': retrieval_x_for_this_client}
        texts = {'query': query_y_for_this_client, 'database': retrieval_y_for_this_client}
        labels = {'query': query_L_for_this_client, 'database': retrieval_L_for_this_client}

        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                   for x in ['query','database']}
        
        shuffle = {'query': False,'database': False}

        dataloader = {x: DataLoader(dataset[x], batch_size=args.batch_size,shuffle=shuffle[x], num_workers=3) for x in ['query','database']}

        testloader = dataloader['query']
        databaseloader = dataloader['database']

        return testloader,databaseloader

    def test_inference(self, idx, args, backbone_list, local_model):
        pass

    def test_inference_twoway(self, idx, args, local_img_model,local_txt_model):
        
        device = args.device
        model_img = local_img_model
        model_txt = local_txt_model

        model_img.to(device)
        model_txt.to(device)

        model_img.eval()
        model_txt.eval()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.databaseloader, self.testloader, model_img,model_txt,device)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        return MAP_I2T, MAP_T2I




if __name__ == '__main__':
    args = args_parser()
    print(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpu)
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_train,dataset_test,client_idcs_train,client_idcs_retrieval,client_idcs_qery = prepare_data_noniid(args=args)


    txt_len = 1386

    private_img_model_list = []
    private_txt_model_list = []


    for _ in range(args.num_users):
      private_model_img =ImageNet(y_dim=4096, bit=args.bit, hiden_layer=args.num_hiden_layers[0])
      private_model_txt =TextNet(y_dim=txt_len, bit=args.bit, hiden_layer=args.num_hiden_layers[1])

      private_img_model_list.append(private_model_img)
      private_txt_model_list.append(private_model_txt)

    global_model_img = ImageNet(y_dim=4096, bit=args.bit, hiden_layer=args.num_hiden_layers[0])
    global_model_txt = TextNet(y_dim=txt_len, bit=args.bit, hiden_layer=args.num_hiden_layers[1])

    global_avg_protos = {}
    local_protos = {}
    best = 0.0

    for round in range(args.rounds):
      print(f'\n | Global Training Round : {round} |\n')
      local_weights_img, local_weights_txt= [],[]
      private_weights_img, private_weights_txt = [],[]
      idxs_users = np.arange(args.num_users)
      for idx in idxs_users:
        print("client",idx)
        local_model = LocalUpdateDJSRH(args=args,id=idx,dataset=dataset_train, idxs=client_idcs_train[idx])

        if len(global_avg_protos) != len(idxs_users):
            global_avg_protos_for_this_client = global_avg_protos
        else:
            global_avg_protos_for_this_client = global_avg_protos[idx]

        w_img,w_txt,private_w_img,private_w_txt,local_centroid= local_model.update_weights_UCCH_new(args,idx,global_avg_protos_for_this_client,model_img=copy.deepcopy(global_model_img), model_txt=copy.deepcopy(global_model_txt),private_img_model=copy.deepcopy(private_img_model_list[idx]), private_txt_model=copy.deepcopy(private_txt_model_list[idx]))

        local_weights_img.append(copy.deepcopy(w_img))
        local_weights_txt.append(copy.deepcopy(w_txt))

        private_weights_img.append(copy.deepcopy(private_w_img))
        private_weights_txt.append(copy.deepcopy(private_w_txt))

        local_protos[idx] = copy.deepcopy(local_centroid)

      local_weights_list_img = average_weights(local_weights_img)
      local_weights_list_txt = average_weights(local_weights_txt)
      global_avg_protos = global_proto_cluster(local_protos)

      for idx in idxs_users:
            private_img_model_list[idx].load_state_dict(private_weights_img[idx])
            private_txt_model_list[idx].load_state_dict(private_weights_txt[idx])
        
      global_model_img = copy.deepcopy(global_model_img)
      global_model_txt = copy.deepcopy(global_model_txt)

      global_model_img.load_state_dict(local_weights_list_img[0], strict=True)
      global_model_txt.load_state_dict(local_weights_list_txt[0], strict=True)


      if round % 2== 0:
          with torch.no_grad():
              MAP_I2T_all, MAP_T2I_all = 0.0,0.0
              for idx in range(args.num_users):
                  local_test = LocalTest(args=args, id = idx,dataloaders=dataset_test,idcs_retrieval = client_idcs_retrieval,idcs_query =client_idcs_qery)
                  local_img_model = copy.deepcopy(global_model_img)
                  local_txt_model = copy.deepcopy(global_model_txt)

                  MAP_I2T, MAP_T2I= local_test.test_inference_twoway(idx, args,local_img_model,local_txt_model)
                  MAP_I2T_all += MAP_I2T
                  MAP_T2I_all += MAP_T2I

              MAP_I2T_ave = MAP_I2T_all / args.num_users
              MAP_T2I_ave = MAP_T2I_all / args.num_users


              print('Test/MAPi2t@50/MAPt2i@50/user' + str(MAP_I2T_ave),MAP_T2I_ave,round)#MAPi2t/MAPt2i/ ,MAP_I2T_a_ave,MAP_T2I_a_ave
              if MAP_T2I_ave + MAP_I2T_ave > best:
                  best = MAP_T2I_ave + MAP_I2T_ave
                  file_name = '%s_%d_bit_latest_our.pth' % (str(args.dataset) , args.bit)

                  ckp_path = osp.join(args.save_model_path, file_name)
                  obj = {
                      'ImgNet': global_model_img.cpu().state_dict(),
                      'TxtNet': global_model_txt.cpu().state_dict(),
                  }
                  torch.save(obj, ckp_path)
                  print('**********Save the trained model successfully.**********')
                  print(best)