To guarantee experimental reproducibility, in this folder, we provide the python source codes and datasets for the KDD submission "Contextualized Self-Supervision for Link Prediction".

This project is built on the open source graph machine learning library StellarGraph. To run this code, please first install StellarGraph from https://github.com/stellargraph/stellargraph

This project is compiled on tensorflow 2.1.0.

Two python programs are provided:

1. "CSSL_context_node.py" performs link prediction using the context node prediction as a self-supervised learning task.

2. "CSSL_context_subgraph.py" performs link prediction using the context subgraph prediction as a self-supervised learning task.

To run "CSSL_context_node.py" on the given four networks, please run the .sh file "run_cssl_context_node.sh", link prediction results with pretraining and joint training will be outputted into the corresponding network folders.

To run "CSSL_context_subgraph.py" on the given four networks, please run the .sh file "run_cssl_context_subgraph.sh", link prediction results with pretraining and joint training will be outputted into the corresponding network folders.
