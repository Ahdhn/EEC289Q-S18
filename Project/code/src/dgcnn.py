from train import trainMain

def dgcnnMain():
    adj_matrix = GenerateImages();
    trainMain(adj_matrix);



dgcnnMain();