import numpy as np
import torch
from metric import evaluate_graph_embeddings_using_svm
from tqdm import tqdm

def train_eval_test_v1(model, optimizer, scheduler, pooler, train_loader, eval_loader, summary_writer, train_test_config):
    n_epoch = train_test_config['max_epoch']
    device = train_test_config['device']
    model = model.to(device)
    # pretrain
    for each_epoch in range(n_epoch):
        model.train()
        
        loss_list = []
        for i,batch in tqdm(enumerate(train_loader),total=len(train_loader),leave=True):
            batch = batch.to(device)
            # print(batch)
            loss, loss_dict = model(batch, batch.x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        print(each_epoch, n_epoch, np.mean(loss_list))
        summary_writer.add_scalar('Pretrain/Loss', np.mean(loss_list), each_epoch)

    torch.save(model.state_dict(), "GAEmodel_cifar_old.pt")
    # model.load_state_dict(torch.load("GAEmodel_audio.pt"))
    # evaluation
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, batch_g in tqdm(enumerate(eval_loader),total=len(eval_loader),leave=True):
            labels = batch_g.y
            batch_g = batch_g.to(device)
            out = model.embed(batch_g, batch_g.x)
            out = pooler(x=out, batch=batch_g.batch)
            # print("这是out",out.shape)
            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1

