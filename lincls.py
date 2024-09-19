# define Linear Classifier for transfer learning
class LinearClassifier(nn.Module):
    def __init__(self, backbone, finetune=True):
        super(LinearClassifier, self).__init__()
        self.backbone = backbone
        dim = self.backbone[1][0].weight.shape[0]
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)
#         self.bn2 = nn.BatchNorm1d(dim)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.fc3 = nn.Linear(dim, 1)

        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        
        if finetune:
            self.ft = finetune
        else:
            self.ft = finetune
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if self.ft:
            x = self.backbone(x)
        else:
            with torch.no_grad():
                x = self.backbone(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.fc3(x)
        return x

def downstream(model, linear_epoch=100):
    loss_hist = {'train':[], 'val':[]}
    start_time = time.time()
    path2weights = './models/lincls_weights.pt'
    max_auc = 0
    auc = 0
    from sklearn import metrics
    linear_scaler = amp.GradScaler()#weight=torch.FloatTensor(class_weight).to(device)
    linear_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([class_weight[1]]).to(device))#nn.CrossEntropyLoss(weight=torch.HalfTensor(class_weight).to(device))
    # linear_opt =  optim.SGD(encoder.parameters(), lr=30, momentum=0.9, weight_decay=1e-4)#pos_weight=torch.FloatTensor(class_weight[1]).to(device)
    linear_opt =  optim.AdamW(model.parameters())
    # start training
    best_result = 0
    for epoch in range(linear_epoch):
        # clear_output()
        print('Epoch {}/{}'.format(epoch+1, linear_epoch))
    
        running_train_loss = 0
        running_val_loss = 0
        running_test_loss = 0
        train_pred, train_gold = [], []
        val_pred, val_gold = [], []
        # transfer dataloader
        model.train()
        for i, (x, y) in enumerate(tqdm(train_dl)):
            # retrieve query and key
            x = x.float().to(device)
            y = y.float().to(device)
            # extract features using linear_encoder
            with amp.autocast():
                pred = model(x)#.reshape(-1,)
                loss = linear_loss_func(pred, y.reshape(-1,1))
                if loss == torch.nan:
                    break
                linear_opt.zero_grad()
                linear_scaler.scale(loss).backward()
                linear_scaler.step(linear_opt)
                linear_scale = linear_scaler.get_scale()
                linear_scaler.update()
            temp_pos = y.int().detach().cpu().numpy()
            train_pred.extend(pred.detach().cpu().numpy())
            train_gold.extend(temp_pos)
            running_train_loss += loss
            
        if loss == torch.nan:
            break
        train_loss = running_train_loss / (i+1)
        loss_hist['train'].append(train_loss.detach().cpu().numpy())
        train_pred, train_gold = np.array(train_pred).reshape(-1), np.array(train_gold).reshape(-1)
        train_AUROC = metrics.roc_auc_score(train_gold, train_pred)
        train_AUPR = metrics.average_precision_score(train_gold, train_pred)
        # validation dataloader
        model.eval()
        for i, (x, y) in enumerate(val_dl):
            x = x.float().to(device)
            y = y.float().to(device)
            
            with torch.no_grad():
                with amp.autocast():
                    pred = model(x)
                    running_val_loss += loss
                val_pred.extend(pred.detach().cpu().numpy())
                val_gold.extend(y.int().detach().cpu().numpy())
        val_pred, val_gold = np.nan_to_num(val_pred).reshape(-1), np.nan_to_num(val_gold).reshape(-1)
        val_AUROC = metrics.roc_auc_score(val_gold, val_pred)
        val_AUPR = metrics.average_precision_score(val_gold, val_pred)
        
        val_loss = running_val_loss / (i+1)
        loss_hist['val'].append(val_loss.detach().cpu().numpy())
        print('train loss: %.6f, val loss: %.6f, AUROC score: %.4f, AUPR score: %.4f, time: %.4f min' %(train_loss, val_loss, val_AUROC, val_AUPR, (time.time()-start_time)/60))
        print('-'*10)
        if val_AUPR > best_result:
            best_loss = val_AUPR
            best_model = copy.deepcopy(model)
    
    best_model.eval()
    test_pred, test_gold = [], []
    for i, (x, y) in enumerate(tqdm(test_dl)):
        # retrieve query and key
        x = x.float().to(device)
        y = y.float().to(device)
        # extract features using q_encoder
        with torch.no_grad():
            with amp.autocast():
                linear_opt.zero_grad()
                pred = best_model(x)# .reshape(-1,)
                loss = linear_loss_func(pred, y.reshape(-1,1))
            test_pred.extend(pred.detach().cpu().numpy())
            test_gold.extend(y.int().detach().cpu().numpy())
    torch.cuda.empty_cache()
    test_loss = running_test_loss / (i+1)
    test_pred, test_gold = np.nan_to_num(test_pred).reshape(-1), np.nan_to_num(test_gold).reshape(-1)
    AUROC = metrics.roc_auc_score(test_gold, test_pred)
    AUPR = metrics.average_precision_score(test_gold, test_pred)
    # F1 = metrics.f1_score(test_gold, test_pred)
    # BAS = metrics.balanced_accuracy_score(test_gold, test_pred)
    # Accuracy = metrics.accuracy_score(test_gold, test_pred)
    print('Test Metric:',AUROC, AUPR)
    return [AUROC, AUPR]