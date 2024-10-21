# professor's code
import itertools
import datetime
import torch
import pubmed_util
import argparse
import create_graph_dgl as cg
import gcnconv_dgl as gnn
import time
import dgl
import torch.nn.functional as F
from dgl.data_utils import load_graphs
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pick CPU or GPU to perfomr GCN')
    parser.add_argument('--gdir', type=str, required=True, help='pick directory to store graph data')
    parser.add_argument('--device', type=str, default='GPU', help='pick device to perform GCN')
    parser.add_argument('-graph', type=str, default='text', help='pick text or binary')
    parser.add_argument('--dim', type=int, default=32, help='intermediate feature dimension of hidden layer')
    parser.add_argument('--category', type=str, required=True, help='classifying category')
    parser.add_argument('--feature', type=str, default='text', help='feature type')
    # add use_ftp16 flag
    parser.add_argument('--use_fp16', action='store_true', help='use this flag to see if we want to use fp16')

    args = parser.parse_args()

    # select device
    use_cuda = args.device == 'GPU' and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print('Using CUDA')
    else:
        print('Using CPU')

    use_fp16 = args.use_fp16
    print("Using half precision: ", use_fp16)

    graph_dir = args.gdir #../data/rtest2/cora
    if args.graph == 'text':
        graph_data_path = graph_dir 
        line_needed_to_skip = 0
        src, dst, comment = cg.read_graph(graph_data_path, line_needed_to_skip)
        graph = cg.build_dgl_graph(src, dst)
    else:
        graph_data_path = graph_dir + '/data.bin'
        graph = load_graphs(graph_data_path)[0][0]
    print(graph)

    # load feature
    feature_type = args.feature
    if feature_type == 'text':
        feature_data_path = graph_dir + '/feature.txt'
        feature = cg.read_feature(feature_data_path)
    else:
        feature_data_path = graph_dir + '/feature.bin'
        feature = torch.load(feature_data_path)

    # load label
    label_data_path = graph_dir + '/label.txt'
    label = cg.read_label(label_data_path)

    # load category
    category = args.category
    category_data_path = graph_dir + '/category.txt'
    category = cg.read_category(category_data_path)

    # create model
    in_dim = feature.size(1)
    out_dim = len(category)
    hidden_dim = args.dim
    model = gnn.GCN(in_dim, hidden_dim, out_dim).to(device)
    print(model)

    # missing code generated
    # Split the dataset into training and testing sets
    num_vcount = graph.number_of_nodes()
    train_mask = torch.zeros(num_vcount, dtype=torch.bool)
    test_mask = torch.zeros(num_vcount, dtype=torch.bool)

    if args.category == 'predefined':
        # Use predefined train/test split
        train_id = pubmed_util.read_label_info(graph_dir + '/train_id.txt')
        test_id = pubmed_util.read_label_info(graph_dir + '/test_id.txt')
        test_y_label = pubmed_util.read_label_info(graph_dir + '/test_y_label.txt')

        train_y_label = pubmed_util.read_label_info(graph_dir + 'label/y_label.txt')   

        train_id = torch.tensor(train_id).to(device)
        test_id = torch
        train_y_label = torch.tensor(train_y_label).to(device)
        test_y_label = torch.tensor(test_y_label).to(device)

    else:
        train = 1
        val = 0.3
        test = 0.1
        num_train = int(num_vcount * val)
        num_test = int(num_vcount * test)
        print("train, test", num_train, num_test, num_vcount)
        train_y_label, test_y_label, train_id, test_id = pubmed_util.ran_init_index_and_label(args.category, num_train, num_test)
        train_y_label = train_y_label.to(device)
        test_y_label = test_y_label.to(device)
        train_id = train_id.to(device)
        test_id = test_id.to(device)

    input_feaure_dim = feature.size(1)
    net = gnn.GCN(input_feature_dim, args.dim, args.category)
    net = net.to(device)

    #train the network
    optimizer = torch.optim.Adam(itertools.chain(net.parameters()), lr=0.01, weight_decay=5e-4) 
    scaler = GradScaler()

    net.train()
    total1 = 0
    start1 = datetime.datetime.now()
    for epoch in range(100):
        start = datetime.datetime.now()
        with autocast(enabled=use_fp16, dtype=torch.float16):
            logits = net(graph, feature)
            loss = F.cross_entropy(logits[train_mask], label[train_mask])
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end = datetime.datetime.now()
        total1 += (end - start).total_seconds()
        print('epoch', epoch, 'loss', loss.item())
    end1 = datetime.datetime.now()
    print('total time', (end1 - start1).total_seconds())
    print('average time', total1 / 100)

    #test the network
    net.eval()
    with torch.no_grad():
        logits = net(graph, feature)
        pred = logits.argmax(1)
        acc = (pred[test_mask] == label[test_mask]).float().mean()
        print('accuracy', acc.item())

    #profile the network
    with profile(activities=[ProfierActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            net(graph, feature)
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=10))
    prof.export_chrome_trace("profile.json")
    print('profile.json is saved')