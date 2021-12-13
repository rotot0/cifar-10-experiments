import torch
from tqdm import tqdm, trange

class Trainer:
    def __init__(self, model, opt, criterion, scheduler=None, device=torch.device('cuda')):
        self.model = model
        self.opt = opt
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        for step, data in enumerate(self.train_loader):
            self.opt.zero_grad()
            imgs = data[0].to(self.device)
            labels = data[1].to(self.device)
            out = self.model(imgs)
            loss = self.criterion(out, labels)
            loss.backward()
            self.opt.step()
            _, preds = out.max(1)
            train_acc += preds.eq(labels).sum()
            train_loss += loss.item()

        if self.scheduler is not None:
                self.scheduler.step()
        return train_loss / len(self.train_loader), train_acc.item() / len(self.train_loader.dataset)
    
    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for step, data in enumerate(self.val_loader):
            imgs = data[0].to(self.device)
            labels = data[1].to(self.device)
            out = self.model(imgs)
            _, preds = out.max(1)
            val_acc += preds.eq(labels).sum()
            loss = self.criterion(out, labels)
            val_loss += loss.item()
        return val_loss / len(self.val_loader), val_acc.item() /  len(self.val_loader.dataset)

    def train(self, train_loader, val_loader, n_epochs):
        print("--> Training is starting...")
        self.train_loader = train_loader
        self.val_loader = val_loader
        for e in range(n_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()
            print(f"Epoch [{e+1}/{n_epochs}] Train loss: {train_loss:.4f} Train Acc: {train_acc:.3f} $ Val loss: {val_loss:.4f} Val Acc: {val_acc:.3f}")
            
        print("--> Training is done...")
        
        return self.model