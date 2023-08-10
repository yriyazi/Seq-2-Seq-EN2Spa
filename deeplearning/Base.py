
import  os
import  torch
import  utils
import  torch.nn    as      nn
import  pandas      as      pd
from    torch.optim import  lr_scheduler
from    tqdm        import  tqdm


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
    
class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def normal_accuracy(predicted,Spa):

    # Acc_over_All = (predicted==Spa).sum()/predicted.shape[0]
    
    
    correct = 0
    num_All_word_exept_pad = torch.count_nonzero(Spa)
    for i in range(len(Spa)):
        if Spa[i] != 0 and Spa[i] == predicted[i]:
            correct += 1
    Acc_with_out_pad = correct / num_All_word_exept_pad
        

    return Acc_with_out_pad*100

def teacher_forcing_ratio(step):
    if step <= 10:
        return max(0.0 - step/150, 0)
    else:
        return 0.0


def accu(Spa,predicted):
    for index in range(Spa.shape[0]):
        Ground = Spa[:,index+1][Spa[:,index+1] != 0][1:-1]
        TTarget = predicted[:,index+1][1:len(Ground)+1]
        acc = (TTarget==Ground).sum()/len(Ground)
    return 100*acc#/Spa.shape[0]

def reformat_tensor(tensor):
    # this method was recommended for parallelize the training by Pytorch
    tensor = tensor.squeeze(dim=1)
    tensor = tensor.transpose(1, 0)
    return tensor

def train(
    train_loader,
    val_loader,
    model,
    model_name,
    epochs,
    learning_rate,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
    
    test_ealuate,
    tets_loader,
    total_iters=20,
    ):

    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.00)#,,reduction='sum'
    # optimzier
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr= learning_rate,
                                 )

    if load_saved_model:
        model, optimizer = load_model(
                                      ckpt_path=ckpt_path, model=model, optimizer=optimizer
                                        )

    lr_schedulerr =  lr_scheduler.LinearLR(optimizer,
                                           start_factor=utils.start_factor,
                                           end_factor=utils.end_factor,
                                           total_iters=total_iters)

    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_acc_nopad_till_current_batch",
            "avg_train_acc_withpad_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_acc_nopad_till_current_batch",
            "avg_val_acc_withpad_till_current_batch",])

    for epoch in tqdm(range(1, epochs + 1)):
        acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()
        mode = "train"
        step = 0
        
        loop_train = tqdm(
                            enumerate(train_loader, 1),
                            total=len(train_loader),
                            desc="train",
                            position=0,
                            leave=True
                        )
        accuracy_withpad_dum = []
        accuracy_nopad_dum   = []
            
        for batch_idx, (En, Spa) in loop_train:
            
            En  = reformat_tensor(En).to(device)
            Spa = reformat_tensor(Spa).to(device)
            
            optimizer.zero_grad()
            output, predicted = model(En, Spa,teacher_forcing_ratio=teacher_forcing_ratio(step))
            step+=1
            Acc = accu(Spa,predicted)
            
            
            output      = output[1:].view(-1, output.shape[-1])
            Spa         = Spa[1:].contiguous().view(-1)
            predicted   = predicted[1:].contiguous().view(-1)
            
            loss        = criterion(output, Spa.to(torch.long))
            # return loss
            
            loss.backward()
            optimizer.step()
            
            accuracy_nopad_dum.append(Acc)
            acc1_accuracy_withpad_dum = sum(accuracy_nopad_dum)/len(accuracy_nopad_dum)

            
            loss_avg_train.update(loss.item(), En.size(0))

            new_row = pd.DataFrame(
                {"model_name": model_name,
                "mode": mode,
                "image_type":"original",
                "epoch": epoch,
                "learning_rate":optimizer.param_groups[0]["lr"],
                "batch_size": En.size(1),
                "batch_index": batch_idx,
                "loss_batch": loss.detach().item(),
                "avg_train_loss_till_current_batch":loss_avg_train.avg,
                "avg_train_acc_nopad_till_current_batch":acc1_accuracy_withpad_dum,
                "avg_train_acc_withpad_till_current_batch":None,#acc1_Acc_over_All ,
                "avg_val_loss_till_current_batch":None,
                "avg_val_acc_nopad_till_current_batch":None,
                "avg_val_acc_withpad_till_current_batch":None },index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                # accuracy_withpad_train="{:.4f}".format(Acc_over_All),
                accuracy_nopad_train="{:.4f}".format(acc1_accuracy_withpad_dum),
                refresh=True,
            )
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )

        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            acc1 = 0
            # total = 0
            accuracy_withpad_dum = []
            accuracy_nopad_dum   = []
            
            for batch_idx, (En, Spa) in loop_val:
                
                En  = reformat_tensor(En).to(device)
                Spa = reformat_tensor(Spa).to(device)
                
                optimizer.zero_grad()
                output, predicted = model(En, Spa,teacher_forcing_ratio=0)
                Acc = accu(Spa,predicted)
            
                output      = output[1:].view(-1, output.shape[-1])
                Spa         = Spa[1:].contiguous().view(-1)
                predicted   = predicted[1:].contiguous().view(-1)
                
                loss        = criterion(output, Spa.to(torch.int64))
                

                accuracy_withpad_dum.append(Acc)
                acc1_accuracy_withpad_dum = sum(accuracy_withpad_dum)/len(accuracy_withpad_dum)
                
                loss_avg_val.update(loss.item(), En.size(0))
                
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                    "mode": mode,
                    "image_type":"original",
                    "epoch": epoch,
                    "learning_rate":optimizer.param_groups[0]["lr"],
                    "batch_size": En.size(1),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch":None,
                    "avg_train_acc_nopad_till_current_batch":None,
                    "avg_train_acc_withpad_till_current_batch":None,
                    "avg_val_loss_till_current_batch":loss_avg_val.avg,
                    "avg_val_acc_nopad_till_current_batch":acc1_accuracy_withpad_dum,
                    "avg_val_acc_withpad_till_current_batch":None,#acc1_Acc_over_All
                    },index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    # accuracy_withpad_train="{:.4f}".format(Acc_over_All),
                    accuracy_nopad_train="{:.4f}".format(acc1_accuracy_withpad_dum),
                    refresh=True,
                )
        if test_ealuate==True:
            mode = "test"
            with torch.no_grad():
                loop_val = tqdm(
                                enumerate(tets_loader, 1),
                                total=len(tets_loader),
                                desc="test",
                                position=0,
                                leave=True,
                                )
                accuracy_withpad_dum = []
                accuracy_nopad_dum   = []
                
                for batch_idx, (En, Spa) in loop_val:
                    En  = reformat_tensor(En).to(device)
                    Spa = reformat_tensor(Spa).to(device)
                    
                    optimizer.zero_grad()
                    output, predicted = model(En, Spa,teacher_forcing_ratio=0)
                
                
                    output      = output[1:].view(-1, output.shape[-1])
                    Spa         = Spa[1:].contiguous().view(-1)
                    predicted   = predicted[1:].contiguous().view(-1)
                    
                    loss        = criterion(output, Spa.to(torch.int64))
                    
                    Acc_over_All , Acc_with_out_pad = normal_accuracy(predicted,Spa)
                    accuracy_nopad_dum.append(Acc_over_All)
                    acc1_Acc_over_All = sum(accuracy_nopad_dum)/len(accuracy_nopad_dum)
                    accuracy_withpad_dum.append(Acc_with_out_pad)
                    acc1_accuracy_withpad_dum = sum(accuracy_withpad_dum)/len(accuracy_withpad_dum)
                    
                    loss_avg_train.update(loss.item(), En.size(0))
                    
                    new_row = pd.DataFrame(
                        {"model_name": model_name,
                    "mode": mode,
                    "image_type":"original",
                    "epoch": epoch,
                    "learning_rate":optimizer.param_groups[0]["lr"],
                    "batch_size": En.size(1),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch":None,
                    "avg_train_acc_nopad_till_current_batch":None,
                    "avg_train_acc_withpad_till_current_batch":None,
                    "avg_val_loss_till_current_batch":loss_avg_val.avg,
                    "avg_val_acc_nopad_till_current_batch":acc1_accuracy_withpad_dum,
                    "avg_val_acc_withpad_till_current_batch":acc1_Acc_over_All
                    },index=[0],)
                    
                    report.loc[len(report)] = new_row.values[0]
                    loop_val.set_description(f"test - iteration : {epoch}")
                    loop_val.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                        accuracy_withpad_train="{:.4f}".format(Acc_over_All),
                        accuracy_nopad_train="{:.4f}".format(acc1_accuracy_withpad_dum),
                        refresh=True,
                    )    
            
        lr_schedulerr.step()
    report.to_csv(f"{report_path}/{model_name}_report.csv")
    torch.save(model.state_dict(), report_path+'/'+model_name+'.pt')
    return model, optimizer, report