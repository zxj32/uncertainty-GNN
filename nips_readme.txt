----------------------------------------------------------------------------------------
This readme.txt describes how to run our proposed method and baselines methods. Before the demonstration, we first take a look at the structure of our code and datasets. The folder structure as following:

	+Uncertainty_Framework
	     +data
                +Amazon.computers
                +Amazon.photo
                +Citeseer
                +Coauthor.Physics
                +Cora
                +Pubmed
	     Baseline.py
	     S-BGCN-T-K.py
	     S-BGCN-T.py
	     S-GCN-T.py
             Kernel_Graph.py
	+nips_readme.txt

    1.  Uncertainty_Framework: It contains the implementation of our methods: S-GCN, S-BGCN,
                         S-BGCN-T, S-BGCN-T-K and baseline: GCN, EDL-GCN, DPN-GCN, Drop-GCN.
    2.  dataset: We consider six datasets: Cora [1], Citeseer [1], Pubmed [1],
                 Amazon.computers[2], Amazon.photo[2], Coauthor.Physics[2]

Our code is written by Python3.6.9 We assume your Operating System is GNU/Linux-based.
However, if you have MacOS or MacBook, it will be okay. If you use Windows, we recommend
that you use PyCharm to run our code. The dependencies of our programs is Python3.6.9.


----------------------------------------------------------------------------------------
This section is to tell you how to prepare the environment. It has three steps:
    1.  install Python3.6.9
    2.  install numpy(1.15.2), tensorflow(1.0 or later) or tensorflow-GUP(1.14.0),
                scikit-learn(0.20.2), scipy(1.1.0).
    (Notice: Our code is based on original GCN code: https://github.com/tkipf/gcn)

After set up above 2 steps, you are ready to run our code. For example,
to get prior Dirichlet Distribution (save in "data/prior/"), please try to run:
    python Kernel_Graph.py

to get pretrain Teacher network (save in "data/output/" or "data/ood/"), please try to run:
    python Baseline.py -dataset cora --model GCN --OOD_detection 1

to run our method (--OOD_detection: 1 means OOD detection task, 0 means Misclassification task):
    python S-BGCN-T-K.py -dataset cora --OOD_detection 1

If there is no error display, then we are done for this section.

----------------------------------------------------------------------------------------
This section describes how to reproduce the results.

To reproduce results of Table-1, run:
	 python S-BGCN-T-K.py -dataset cora --OOD_detection 0
     python Baseline.py -dataset cora --model EDL --OOD_detection 0
     python Baseline.py -dataset cora --model DPN --OOD_detection 0
     python Baseline.py -dataset cora --model Drop --OOD_detection 0
     python Baseline.py -dataset cora --model GCN --OOD_detection 0

To reproduce results of Table-2, run:
	 python S-BGCN-T-K.py -dataset cora --OOD_detection 1
     python Baseline.py -dataset cora --model EDL --OOD_detection 1
     python Baseline.py -dataset cora --model DPN --OOD_detection 1
     python Baseline.py -dataset cora --model Drop --OOD_detection 1
     python Baseline.py -dataset cora --model GCN --OOD_detection 1

To reproduce results of Table-3, run:
	 python S-BGCN-T-K.py -dataset cora --OOD_detection 0
	 python S-BGCN-T.py -dataset cora --OOD_detection 0
	 python S-GCN.py -dataset cora  --model S-BGCN --OOD_detection 0
	 python S-GCN.py -dataset cora  --model S-GCN --OOD_detection 0

	 python S-BGCN-T-K_npz.py -dataset amazon_electronics_photo --OOD_detection 0
	 python S-BGCN-T_npz.py -dataset amazon_electronics_photo --OOD_detection 0
	 python S-GCN_npz.py -dataset amazon_electronics_photo  --model S-BGCN --OOD_detection 0
	 python S-GCN_npz.py -dataset amazon_electronics_photo  --model S-GCN --OOD_detection 0

Notice: Some programs above are time-cost if you only use one cpu. A better way is to
test them in GPU. After above steps, you should be able to reproduce our results
reported  in our paper. If you cannot reproduce, please email: --@--.

References:
[1] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and
		Tina Eliassi-Rad. Collective classification in network data. AI magazine, 29(3):93,
		2008.
[2] O. Shchur, M. Mumme, A. Bojchevski, and S. Günnemann. Pitfalls of graph neural
    networkevaluation.arXiv preprint arXiv:1811.05868, 2018.

----------------------------------------------------------------------------------------