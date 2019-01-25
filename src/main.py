from sgcn import SignedGCNTrainer
from parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs

def main():
    """
    Parsing command line parameters, creating target matrix, fitting an SGCN, predicting edge signs, and saving the embedding.
    """
    args = parameter_parser()
    tab_printer(args)
    # edges = read_graph(args)
    edges, nodes_dict = read_graph(args) # nodes_dict['indice']:node_id , nodes_dict['label'] : label
    trainer = SignedGCNTrainer(args, edges, nodes_dict)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

if __name__ == "__main__":
    main()
