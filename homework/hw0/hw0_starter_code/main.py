import argparse

from dataset import get_movielens_dataset
from evaluation import mrr_score, mse_score
from models import MultiTaskNet
from multitask import MultitaskModel
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

def main(config):
    print(config)
    # writer = SummaryWriter(config.logdir)

    dataset = get_movielens_dataset(variant='100K')
    train, test = dataset.random_train_test_split(test_fraction=config.test_fraction)

    net = MultiTaskNet(train.num_users,
                       train.num_items,
                       embedding_sharing=config.shared_embeddings)
    model = MultitaskModel(interactions=train,
                           representation=net,
                           factorization_weight=config.factorization_weight,
                           regression_weight=config.regression_weight)
    factorization = []
    score = []
    joint = []
    eval_mse = []
    rank = []
    for epoch in range(config.epochs):
        factorization_loss, score_loss, joint_loss = model.fit(train)
        mrr = mrr_score(model, test, train)
        mse = mse_score(model, test)
        print(f"training/Factorization Loss : {factorization_loss} , epoch {epoch}")
        factorization.append((epoch,factorization_loss))
        print(f"training/MSE : {score_loss} , epoch {epoch}")
        score.append((epoch , score_loss))
        print(f"training/Joint Loss : {joint_loss} , epoch {epoch}")
        joint.append((epoch,joint_loss))
        print(f"eval/Mean Reciprocal Rank : {mrr} , epoch {epoch}")
        rank.append((epoch , mrr))
        print(f"eval/MSE : {mse} , epoch {epoch}")
        eval_mse.append((epoch , mse))

        # writer.add_scalar('training/Factorization Loss', factorization_loss, epoch)
        # writer.add_scalar('training/MSE', score_loss, epoch)
        # writer.add_scalar('training/Joint Loss', joint_loss, epoch)
        # writer.add_scalar('eval/Mean Reciprocal Rank', mrr, epoch)
        # writer.add_scalar('eval/MSE', mse, epoch)
    x , y = zip(*factorization)
    plt.plot(x,y,color='red')
    plt.title("Factorization Loss")
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig("factorization_loss.png")
    plt.show()
    x , y = zip(*score)
    plt.plot(x,y,color='red')
    plt.title("Score Loss")
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig("score_loss.png")
    plt.show()
    x , y = zip(*joint)
    plt.plot(x,y,color='red')
    plt.title("Joint Loss")
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig("joint_loss.png")
    plt.show()
    x , y = zip(*eval_mse)
    plt.plot(x,y,color='red')
    plt.title("MSE")
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig("mse.png")
    plt.show()
    x , y = zip(*rank)
    plt.plot(x,y,color='red')
    plt.title("Mean Reciprocal Rank")
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig("rank.png")
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_fraction', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--factorization_weight', type=float, default=0.995)
    parser.add_argument('--regression_weight', type=float, default=0.005)
    parser.add_argument('--shared_embeddings', default=True,
                        action='store_true')
    parser.add_argument('--no_shared_embeddings', dest='shared_embeddings',
                        action='store_false')
    parser.add_argument('--logdir', type=str,
                        default='run/shared=True_LF=0.99_LR=0.01')
    main(parser.parse_args())