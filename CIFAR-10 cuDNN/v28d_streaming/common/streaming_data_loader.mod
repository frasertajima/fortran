V35 :0x24 streaming_data_loader
25 streaming_data_loader.cuf S624 0
11/21/2025  13:19:37
use iso_c_binding public 0 indirect
use nvf_acc_common public 0 indirect
use cuda_runtime_api public 0 indirect
use cutensor_v2_types public 0 indirect
use cutensor_v2 public 0 indirect
use cutensorex_types public 0 indirect
use gpu_reduc_types public 0 indirect
use gpu_reductions public 0 indirect
use sort public 0 indirect
use cudafor_la public 0 direct
use omp_lib_kinds public 0 indirect
use omp_lib public 0 direct
enduse
D 58 26 647 8 646 7
D 67 26 650 8 649 7
D 76 26 647 8 646 7
D 97 26 744 8 743 7
D 378 26 1703 88 1702 7
D 384 23 7 1 11 705 0 0 0 0 0
 0 705 11 11 705 705
D 1493 26 7767 4 7766 3
D 1502 26 7780 4 7779 3
D 1511 26 7789 4 7788 3
D 1520 26 7858 4 7857 3
D 1529 26 7891 4 7890 3
D 1538 26 7950 4 7949 3
D 1547 26 7973 4 7972 3
D 1556 26 7990 4 7989 3
D 1565 26 8005 4 8004 3
D 1574 26 8010 4 8009 3
D 1583 26 8017 4 8016 3
D 1592 26 8024 4 8023 3
D 4115 26 11996 360 11995 7
D 4121 23 9 2 4415 4414 0 1 0 0 1
 4404 4407 4412 4404 4407 4405
 4408 4411 4413 4408 4411 4409
D 4124 23 7 1 0 1452 0 0 0 0 0
 0 1452 0 11 1452 0
D 4127 23 6 1 4424 4423 0 1 0 0 1
 4418 4421 4422 4418 4421 4419
D 4130 23 7 1 0 1261 0 0 0 0 0
 0 1261 0 11 1261 0
D 4133 22 7
D 4135 22 7
D 4137 23 7 1 0 11 0 0 0 0 0
 0 11 0 11 11 0
D 4140 23 7 1 0 4427 0 0 0 0 0
 0 4427 0 11 4427 0
D 4143 23 7 1 0 11 0 0 0 0 0
 0 11 0 11 11 0
D 4146 23 7 1 0 4427 0 0 0 0 0
 0 4427 0 11 4427 0
D 4149 23 7 1 0 11 0 0 0 0 0
 0 11 0 11 11 0
D 4152 20 1987
D 4154 23 6 1 4441 4440 0 1 0 0 1
 4435 4438 4439 4435 4438 4436
D 4157 23 7 1 0 706 0 0 0 0 0
 0 706 0 11 706 0
D 4164 23 9 2 4447 4453 1 1 0 0 1
 11 4448 11 11 4448 4449
 11 4450 4451 11 4450 4452
D 4167 23 6 1 4454 4457 1 1 0 0 1
 11 4455 11 11 4455 4456
S 624 24 0 0 0 9 1 0 4986 10005 8000 A 0 0 0 0 B 0 28 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 28 0 0 0 0 0 0 streaming_data_loader
S 629 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 630 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 631 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
R 646 25 7 iso_c_binding c_ptr
R 647 5 8 iso_c_binding val c_ptr
R 649 25 10 iso_c_binding c_funptr
R 650 5 11 iso_c_binding val c_funptr
R 684 6 45 iso_c_binding c_null_ptr$ac
R 686 6 47 iso_c_binding c_null_funptr$ac
R 687 26 48 iso_c_binding ==
R 689 26 50 iso_c_binding !=
S 715 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 716 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 717 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 718 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 719 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 720 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 721 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 722 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 19 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 723 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 724 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 725 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 726 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 20 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 727 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 21 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 728 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 729 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 23 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 730 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 731 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 732 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 733 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 734 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 24 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 735 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 736 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 26 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 737 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 27 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
R 743 25 6 nvf_acc_common c_devptr
R 744 5 7 nvf_acc_common cptr c_devptr
R 750 6 13 nvf_acc_common c_null_devptr$ac
R 788 26 51 nvf_acc_common =
S 848 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 28 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 849 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 31 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 850 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 851 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 35 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 852 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 37 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 934 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 935 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 29 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 936 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 30 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 937 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 33 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 938 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 34 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 939 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 36 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 940 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 38 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 941 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 39 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 942 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 40 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 943 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 41 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 944 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 42 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 947 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 50 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 959 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 1007 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 126 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 1008 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 1026 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1 -2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 1027 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 1032 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1033 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1034 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1044 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1045 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1046 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1049 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1050 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1051 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 21 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 1056 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 23 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
R 1702 25 626 cuda_runtime_api cudapointerattributes
R 1703 5 627 cuda_runtime_api type cudapointerattributes
R 1704 5 628 cuda_runtime_api device cudapointerattributes
R 1705 5 629 cuda_runtime_api devptr cudapointerattributes
R 1706 5 630 cuda_runtime_api hostptr cudapointerattributes
R 1707 5 631 cuda_runtime_api reserved cudapointerattributes
S 7753 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1 -6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 7754 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1 -4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 7755 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1 -3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 7756 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 1024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 7757 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 4096 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 7758 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 8192 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 7759 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 512 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
R 7766 25 1 cutensor_v2_types cutensoralgo
R 7767 5 2 cutensor_v2_types algo cutensoralgo
R 7770 6 5 cutensor_v2_types cutensor_algo_default_patient$ac
R 7772 6 7 cutensor_v2_types cutensor_algo_gett$ac
R 7774 6 9 cutensor_v2_types cutensor_algo_tgett$ac
R 7776 6 11 cutensor_v2_types cutensor_algo_ttgt$ac
R 7778 6 13 cutensor_v2_types cutensor_algo_default$ac
R 7779 25 14 cutensor_v2_types cutensorworksizepreference
R 7780 5 15 cutensor_v2_types wksp cutensorworksizepreference
R 7783 6 18 cutensor_v2_types cutensor_workspace_min$ac
R 7785 6 20 cutensor_v2_types cutensor_workspace_default$ac
R 7787 6 22 cutensor_v2_types cutensor_workspace_max$ac
R 7788 25 23 cutensor_v2_types cutensoroperator
R 7789 5 24 cutensor_v2_types opno cutensoroperator
R 7792 6 27 cutensor_v2_types cutensor_op_identity$ac
R 7794 6 29 cutensor_v2_types cutensor_op_sqrt$ac
R 7796 6 31 cutensor_v2_types cutensor_op_relu$ac
R 7798 6 33 cutensor_v2_types cutensor_op_conj$ac
R 7800 6 35 cutensor_v2_types cutensor_op_rcp$ac
R 7802 6 37 cutensor_v2_types cutensor_op_sigmoid$ac
R 7804 6 39 cutensor_v2_types cutensor_op_tanh$ac
R 7806 6 41 cutensor_v2_types cutensor_op_exp$ac
R 7808 6 43 cutensor_v2_types cutensor_op_log$ac
R 7810 6 45 cutensor_v2_types cutensor_op_abs$ac
R 7812 6 47 cutensor_v2_types cutensor_op_neg$ac
R 7814 6 49 cutensor_v2_types cutensor_op_sin$ac
R 7816 6 51 cutensor_v2_types cutensor_op_cos$ac
R 7818 6 53 cutensor_v2_types cutensor_op_tan$ac
R 7820 6 55 cutensor_v2_types cutensor_op_sinh$ac
R 7822 6 57 cutensor_v2_types cutensor_op_cosh$ac
R 7824 6 59 cutensor_v2_types cutensor_op_asin$ac
R 7826 6 61 cutensor_v2_types cutensor_op_acos$ac
R 7828 6 63 cutensor_v2_types cutensor_op_atan$ac
R 7830 6 65 cutensor_v2_types cutensor_op_asinh$ac
R 7832 6 67 cutensor_v2_types cutensor_op_acosh$ac
R 7834 6 69 cutensor_v2_types cutensor_op_atanh$ac
R 7836 6 71 cutensor_v2_types cutensor_op_ceil$ac
R 7838 6 73 cutensor_v2_types cutensor_op_floor$ac
R 7840 6 75 cutensor_v2_types cutensor_op_mish$ac
R 7842 6 77 cutensor_v2_types cutensor_op_swish$ac
R 7844 6 79 cutensor_v2_types cutensor_op_soft_plus$ac
R 7846 6 81 cutensor_v2_types cutensor_op_soft_sign$ac
R 7848 6 83 cutensor_v2_types cutensor_op_add$ac
R 7850 6 85 cutensor_v2_types cutensor_op_mul$ac
R 7852 6 87 cutensor_v2_types cutensor_op_max$ac
R 7854 6 89 cutensor_v2_types cutensor_op_min$ac
R 7856 6 91 cutensor_v2_types cutensor_op_unknown$ac
R 7857 25 92 cutensor_v2_types cutensorstatus
R 7858 5 93 cutensor_v2_types stat cutensorstatus
R 7861 6 96 cutensor_v2_types cutensor_status_success$ac
R 7863 6 98 cutensor_v2_types cutensor_status_not_initialized$ac
R 7865 6 100 cutensor_v2_types cutensor_status_alloc_failed$ac
R 7867 6 102 cutensor_v2_types cutensor_status_invalid_value$ac
R 7869 6 104 cutensor_v2_types cutensor_status_arch_mismatch$ac
R 7871 6 106 cutensor_v2_types cutensor_status_mapping_error$ac
R 7873 6 108 cutensor_v2_types cutensor_status_execution_failed$ac
R 7875 6 110 cutensor_v2_types cutensor_status_internal_error$ac
R 7877 6 112 cutensor_v2_types cutensor_status_not_supported$ac
R 7879 6 114 cutensor_v2_types cutensor_status_license_error$ac
R 7881 6 116 cutensor_v2_types cutensor_status_cublas_error$ac
R 7883 6 118 cutensor_v2_types cutensor_status_cuda_error$ac
R 7885 6 120 cutensor_v2_types cutensor_status_insufficient_workspace$ac
R 7887 6 122 cutensor_v2_types cutensor_status_insufficient_driver$ac
R 7889 6 124 cutensor_v2_types cutensor_status_io_error$ac
R 7890 25 125 cutensor_v2_types cutensordatatype
R 7891 5 126 cutensor_v2_types cudadatatype cutensordatatype
R 7894 6 129 cutensor_v2_types cutensor_r_16f$ac
R 7896 6 131 cutensor_v2_types cutensor_c_16f$ac
R 7898 6 133 cutensor_v2_types cutensor_r_16bf$ac
R 7900 6 135 cutensor_v2_types cutensor_c_16bf$ac
R 7902 6 137 cutensor_v2_types cutensor_r_32f$ac
R 7904 6 139 cutensor_v2_types cutensor_c_32f$ac
R 7906 6 141 cutensor_v2_types cutensor_r_64f$ac
R 7908 6 143 cutensor_v2_types cutensor_c_64f$ac
R 7910 6 145 cutensor_v2_types cutensor_r_4i$ac
R 7912 6 147 cutensor_v2_types cutensor_c_4i$ac
R 7914 6 149 cutensor_v2_types cutensor_r_4u$ac
R 7916 6 151 cutensor_v2_types cutensor_c_4u$ac
R 7918 6 153 cutensor_v2_types cutensor_r_8i$ac
R 7920 6 155 cutensor_v2_types cutensor_c_8i$ac
R 7922 6 157 cutensor_v2_types cutensor_r_8u$ac
R 7924 6 159 cutensor_v2_types cutensor_c_8u$ac
R 7926 6 161 cutensor_v2_types cutensor_r_16i$ac
R 7928 6 163 cutensor_v2_types cutensor_c_16i$ac
R 7930 6 165 cutensor_v2_types cutensor_r_16u$ac
R 7932 6 167 cutensor_v2_types cutensor_c_16u$ac
R 7934 6 169 cutensor_v2_types cutensor_r_32i$ac
R 7936 6 171 cutensor_v2_types cutensor_c_32i$ac
R 7938 6 173 cutensor_v2_types cutensor_r_32u$ac
R 7940 6 175 cutensor_v2_types cutensor_c_32u$ac
R 7942 6 177 cutensor_v2_types cutensor_r_64i$ac
R 7944 6 179 cutensor_v2_types cutensor_c_64i$ac
R 7946 6 181 cutensor_v2_types cutensor_r_64u$ac
R 7948 6 183 cutensor_v2_types cutensor_c_64u$ac
R 7949 25 184 cutensor_v2_types cutensorcomputetype
R 7950 5 185 cutensor_v2_types type cutensorcomputetype
R 7953 6 188 cutensor_v2_types cutensor_compute_16f$ac
R 7955 6 190 cutensor_v2_types cutensor_compute_16bf$ac
R 7957 6 192 cutensor_v2_types cutensor_compute_tf32$ac
R 7959 6 194 cutensor_v2_types cutensor_compute_3xtf32$ac
R 7961 6 196 cutensor_v2_types cutensor_compute_32f$ac
R 7963 6 198 cutensor_v2_types cutensor_compute_64f$ac
R 7965 6 200 cutensor_v2_types cutensor_compute_8u$ac
R 7967 6 202 cutensor_v2_types cutensor_compute_8i$ac
R 7969 6 204 cutensor_v2_types cutensor_compute_32u$ac
R 7971 6 206 cutensor_v2_types cutensor_compute_32i$ac
R 7972 25 207 cutensor_v2_types cutensoroperationdescriptorattribute
R 7973 5 208 cutensor_v2_types attr cutensoroperationdescriptorattribute
R 7976 6 211 cutensor_v2_types cutensor_operation_descriptor_tag$ac
R 7978 6 213 cutensor_v2_types cutensor_operation_descriptor_scalar_type$ac
R 7980 6 215 cutensor_v2_types cutensor_operation_descriptor_flops$ac
R 7982 6 217 cutensor_v2_types cutensor_operation_descriptor_moved_bytes$ac
R 7984 6 219 cutensor_v2_types cutensor_operation_descriptor_padding_left$ac
R 7986 6 221 cutensor_v2_types cutensor_operation_descriptor_padding_right$ac
R 7988 6 223 cutensor_v2_types cutensor_operation_descriptor_padding_value$ac
R 7989 25 224 cutensor_v2_types cutensorplanpreferenceattribute
R 7990 5 225 cutensor_v2_types attr cutensorplanpreferenceattribute
R 7993 6 228 cutensor_v2_types cutensor_plan_preference_autotune_mode$ac
R 7995 6 230 cutensor_v2_types cutensor_plan_preference_cache_mode$ac
R 7997 6 232 cutensor_v2_types cutensor_plan_preference_incremental_count$ac
R 7999 6 234 cutensor_v2_types cutensor_plan_preference_algo$ac
R 8001 6 236 cutensor_v2_types cutensor_plan_preference_kernel_rank$ac
R 8003 6 238 cutensor_v2_types cutensor_plan_preference_jit$ac
R 8004 25 239 cutensor_v2_types cutensorplanattribute
R 8005 5 240 cutensor_v2_types attr cutensorplanattribute
R 8008 6 243 cutensor_v2_types cutensor_plan_required_workspace$ac
R 8009 25 244 cutensor_v2_types cutensorautotunemode
R 8010 5 245 cutensor_v2_types mode cutensorautotunemode
R 8013 6 248 cutensor_v2_types cutensor_autotune_mode_none$ac
R 8015 6 250 cutensor_v2_types cutensor_autotune_mode_incremental$ac
R 8016 25 251 cutensor_v2_types cutensorjitmode
R 8017 5 252 cutensor_v2_types mode cutensorjitmode
R 8020 6 255 cutensor_v2_types cutensor_jit_mode_none$ac
R 8022 6 257 cutensor_v2_types cutensor_jit_mode_default$ac
R 8023 25 258 cutensor_v2_types cutensorcachemode
R 8024 5 259 cutensor_v2_types mode cutensorcachemode
R 8027 6 262 cutensor_v2_types cutensor_cache_mode_none$ac
R 8029 6 264 cutensor_v2_types cutensor_cache_mode_pedantic$ac
R 9145 26 436 gpu_reductions *
S 11988 16 0 0 0 6 1 624 84499 4 400000 A 0 0 0 0 B 0 38 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 stream_mode_disabled
S 11989 16 0 0 0 6 1 624 84520 4 400000 A 0 0 0 0 B 0 39 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 stream_mode_enabled
S 11990 16 0 0 0 6 1 624 84540 4 400000 A 0 0 0 0 B 0 40 0 0 0 0 0 0 2 15 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 stream_mode_auto
S 11991 16 0 0 0 6 1 624 84557 4 400000 A 0 0 0 0 B 0 43 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 shuffle_none
S 11992 16 1 0 0 6 1 624 84570 4 400000 A 0 0 0 0 B 0 44 0 0 0 0 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 shuffle_block
S 11993 16 0 0 0 6 1 624 84584 4 400000 A 0 0 0 0 B 0 45 0 0 0 0 0 0 2 15 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 shuffle_full
S 11994 16 1 0 0 6 1 624 84597 4 400000 A 0 0 0 0 B 0 48 0 0 0 0 0 0 50 465 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 default_block_size
S 11995 25 0 0 0 4115 1 624 84616 1000000c 800050 A 0 0 0 0 B 0 53 0 0 0 0 0 0 0 0 0 12012 0 0 0 0 0 0 0 12011 0 0 0 624 0 0 0 0 stream_buffer_t
S 11996 5 6 0 0 4121 12000 624 29206 10a00004 51 A 0 0 0 0 B 1 54 0 0 0 0 12000 0 4115 0 12002 0 0 0 0 0 0 0 0 11999 1 11996 12001 624 0 0 0 0 data
S 11997 6 4 0 0 7 11998 624 22919 40800006 0 A 0 0 0 0 B 0 54 0 0 0 0 0 0 0 0 0 0 12056 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_0
S 11998 6 4 0 0 7 12004 624 6530 40800006 0 A 0 0 0 0 B 0 54 0 0 0 8 0 0 0 0 0 0 12056 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_1
S 11999 5 1 0 0 4124 12003 624 84632 40822004 1020 A 0 0 0 0 B 0 54 0 0 0 16 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 12001 11999 0 624 0 0 0 0 data$sd
S 12000 5 0 0 0 7 12001 624 84640 40802001 1020 A 0 0 0 0 B 0 54 0 0 0 0 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 11996 12000 0 624 0 0 0 0 data$p
S 12001 5 0 0 0 7 11999 624 84647 40802000 1020 A 0 0 0 0 B 0 54 0 0 0 8 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 12000 12001 0 624 0 0 0 0 data$o
S 12002 22 1 0 0 9 1 624 84654 40000000 1000 A 0 0 0 0 B 0 54 0 0 0 0 0 11996 0 0 0 0 11999 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 data$arrdsc
S 12003 5 6 0 0 4127 12006 624 84666 10a00004 51 A 0 0 0 0 B 1 55 0 0 0 200 12006 0 4115 0 12008 0 0 0 0 0 0 0 0 12005 11996 12003 12007 624 0 0 0 0 labels
S 12004 6 4 0 0 7 12039 624 6536 40800006 0 A 0 0 0 0 B 0 55 0 0 0 16 0 0 0 0 0 0 12056 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_2
S 12005 5 1 0 0 4130 12009 624 84673 40822004 1020 A 0 0 0 0 B 0 55 0 0 0 216 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 12007 12005 0 624 0 0 0 0 labels$sd
S 12006 5 0 0 0 7 12007 624 84683 40802001 1020 A 0 0 0 0 B 0 55 0 0 0 200 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 12003 12006 0 624 0 0 0 0 labels$p
S 12007 5 0 0 0 7 12005 624 84692 40802000 1020 A 0 0 0 0 B 0 55 0 0 0 208 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 12006 12007 0 624 0 0 0 0 labels$o
S 12008 22 1 0 0 6 1 624 84701 40000000 1000 A 0 0 0 0 B 0 55 0 0 0 0 0 12003 0 0 0 0 12005 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 labels$arrdsc
S 12009 5 0 0 0 6 12010 624 84715 800004 0 A 0 0 0 0 B 0 56 0 0 0 352 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 12003 12009 0 624 0 0 0 0 batch_idx
S 12010 5 0 0 0 18 1 624 84725 800004 0 A 0 0 0 0 B 0 57 0 0 0 356 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 12009 12010 0 624 0 0 0 0 ready
S 12011 8 5 0 0 4137 1 624 84731 40822004 1220 A 0 0 0 0 B 0 58 0 0 0 0 0 4115 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_data_loader$$$stream_buffer_t$$$td
S 12012 6 4 0 0 4115 1 624 84776 80004e 0 A 0 0 0 0 B 800 58 0 0 0 0 0 0 0 0 0 0 12057 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 ._dtInit4115
S 12013 6 4 0 0 4115 12014 624 84789 14 8 A 0 0 0 0 B 0 65 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 buffer_a
S 12014 6 4 0 0 4115 12031 624 84798 14 8 A 0 0 0 0 B 0 65 0 0 0 360 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 buffer_b
S 12015 6 6 0 0 4115 1 624 84807 80001c 14 A 0 0 0 0 B 0 66 0 0 0 0 12019 0 0 0 0 0 0 0 0 0 0 0 0 12018 0 0 12020 624 0 0 0 0 current_buffer
S 12016 3 0 0 0 7 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 12017 8 1 0 0 4140 1 624 84822 40822006 1020 A 0 0 0 0 B 0 66 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 current_buffer$sd
S 12018 8 4 0 0 4143 12024 624 84840 40822014 1020 A 0 0 0 0 B 0 66 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 current_buffer$sd1
S 12019 6 4 0 0 7 12020 624 84859 40802011 1020 A 0 0 0 0 B 0 66 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 current_buffer$p
S 12020 6 4 0 0 7 12018 624 84876 40802010 1020 A 0 0 0 0 B 0 66 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 current_buffer$o
S 12021 6 6 0 0 4115 1 624 84893 80001c 14 A 0 0 0 0 B 0 67 0 0 0 0 12024 0 0 0 0 0 0 0 0 0 0 0 0 12023 0 0 12025 624 0 0 0 0 loading_buffer
S 12022 8 1 0 0 4146 1 624 84908 40822006 1020 A 0 0 0 0 B 0 67 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 loading_buffer$sd
S 12023 8 4 0 0 4149 12041 624 84926 40822014 1020 A 0 0 0 0 B 0 67 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 loading_buffer$sd2
S 12024 6 4 0 0 7 12025 624 84945 40802011 1020 A 0 0 0 0 B 0 67 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 loading_buffer$p
S 12025 6 4 0 0 7 12023 624 84962 40802010 1020 A 0 0 0 0 B 0 67 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 loading_buffer$o
S 12026 6 4 0 0 6 12027 624 84979 80001c 0 A 0 0 0 0 B 0 70 0 0 0 0 0 0 0 0 0 0 12059 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 data_file_unit
S 12027 6 4 0 0 6 12044 624 84994 80001c 0 A 0 0 0 0 B 0 71 0 0 0 4 0 0 0 0 0 0 12059 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 label_file_unit
S 12028 6 4 0 0 4152 12029 624 85010 14 0 A 0 0 0 0 B 0 72 0 0 0 0 0 0 0 0 0 0 12060 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 data_filename
S 12029 6 4 0 0 4152 1 624 85024 14 0 A 0 0 0 0 B 0 73 0 0 0 512 0 0 0 0 0 0 12060 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 label_filename
S 12030 6 4 0 0 7 12033 624 85039 14 0 A 0 0 0 0 B 0 76 0 0 0 0 0 0 0 0 0 0 12061 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 total_samples
S 12031 6 4 0 0 6 12032 624 85053 14 0 A 0 0 0 0 B 0 77 0 0 0 720 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 batch_size
S 12032 6 4 0 0 6 12036 624 85064 14 0 A 0 0 0 0 B 0 78 0 0 0 724 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 feature_size
S 12033 6 4 0 0 7 12034 624 85077 14 0 A 0 0 0 0 B 0 79 0 0 0 8 0 0 0 0 0 0 12061 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 bytes_per_sample
S 12034 6 4 0 0 7 12035 624 85094 14 0 A 0 0 0 0 B 0 80 0 0 0 16 0 0 0 0 0 0 12061 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 bytes_per_batch
S 12035 6 4 0 0 7 1 624 85110 14 0 A 0 0 0 0 B 0 81 0 0 0 24 0 0 0 0 0 0 12061 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 label_bytes_per_batch
S 12036 6 4 0 0 6 12037 624 85132 14 0 A 0 0 0 0 B 0 82 0 0 0 728 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 total_batches
S 12037 6 4 0 0 6 1 624 85146 14 0 A 0 0 0 0 B 0 85 0 0 0 732 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 current_batch_idx
S 12038 7 6 0 0 4154 1 624 85164 10a00014 51 A 0 0 0 0 B 0 86 0 0 0 0 12041 0 0 0 12043 0 0 0 0 0 0 0 0 12040 0 0 12042 624 0 0 0 0 batch_order
S 12039 6 4 0 0 7 1 624 6542 40800006 0 A 0 0 0 0 B 0 86 0 0 0 24 0 0 0 0 0 0 12056 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_3
S 12040 8 4 0 0 4157 12013 624 85176 40822014 1020 A 0 0 0 0 B 0 86 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 batch_order$sd
S 12041 6 4 0 0 7 12042 624 85191 40802011 1020 A 0 0 0 0 B 0 86 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 batch_order$p
S 12042 6 4 0 0 7 12040 624 85205 40802010 1020 A 0 0 0 0 B 0 86 0 0 0 0 0 0 0 0 0 0 12058 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 batch_order$o
S 12043 22 1 0 0 9 1 624 85219 40000000 1000 A 0 0 0 0 B 0 86 0 0 0 0 0 12038 0 0 0 0 12040 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 batch_order$arrdsc
S 12044 6 4 0 0 6 12045 624 85238 80001c 0 A 0 0 0 0 B 0 89 0 0 0 8 0 0 0 0 0 0 12059 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 shuffle_mode
S 12045 6 4 0 0 6 12046 624 85251 80001c 0 A 0 0 0 0 B 0 90 0 0 0 12 0 0 0 0 0 0 12059 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 block_size
S 12046 6 4 0 0 18 12048 624 85262 80001c 0 A 0 0 0 0 B 0 91 0 0 0 16 0 0 0 0 0 0 12059 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 is_initialized
S 12048 6 4 0 0 18 1 624 85277 80001c 0 A 0 0 0 0 B 0 92 0 0 0 20 0 0 0 0 0 0 12059 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 files_open
S 12049 27 0 0 0 9 12062 624 85288 0 8000000 A 0 0 0 0 B 0 97 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_init
S 12050 27 0 0 0 9 12069 624 85303 0 8000000 A 0 0 0 0 B 0 98 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_start_epoch
S 12051 27 0 0 0 9 12071 624 85325 0 8000000 A 0 0 0 0 B 0 99 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_get_batch
S 12052 27 0 0 0 9 12101 624 85345 0 8000000 A 0 0 0 0 B 0 100 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_cleanup
S 12053 27 0 0 0 9 12091 624 85363 0 8000000 A 0 0 0 0 B 0 101 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_set_shuffle_mode
S 12054 27 0 0 0 9 12095 624 85390 0 8000000 A 0 0 0 0 B 0 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_get_num_batches
S 12055 27 0 0 0 9 12098 624 85416 0 8000000 A 0 0 0 0 B 0 103 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 streaming_is_initialized
S 12056 11 0 0 0 9 8057 624 85441 40800000 805000 A 0 0 0 0 B 0 105 0 0 0 32 0 0 11997 12039 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _streaming_data_loader$2
S 12057 11 0 0 0 9 12056 624 85466 40800000 805000 A 0 0 0 0 B 0 105 0 0 0 360 0 0 12012 12012 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _streaming_data_loader$8
S 12058 11 0 0 0 9 12057 624 85491 40800010 805000 A 0 0 0 0 B 0 105 0 0 0 928 0 0 12019 12037 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _streaming_data_loader$4
S 12059 11 0 0 0 9 12058 624 85516 40800010 805000 A 0 0 0 0 B 0 105 0 0 0 24 0 0 12026 12048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _streaming_data_loader$12
S 12060 11 0 0 0 9 12059 624 85542 40800010 805000 A 0 0 0 0 B 0 105 0 0 0 1024 0 0 12028 12029 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _streaming_data_loader$5
S 12061 11 0 0 0 9 12060 624 85567 40800010 805000 A 0 0 0 0 B 0 105 0 0 0 32 0 0 12030 12035 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _streaming_data_loader$6
S 12062 23 5 0 0 0 12068 624 85288 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 streaming_init
S 12063 1 3 1 0 30 1 12062 85592 4 43000 A 0 0 0 0 B 0 110 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 data_file
S 12064 1 3 1 0 30 1 12062 85602 4 43000 A 0 0 0 0 B 0 110 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 label_file
S 12065 1 3 1 0 7 1 12062 85613 4 3000 A 0 0 0 0 B 0 110 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 num_samples
S 12066 1 3 1 0 6 1 12062 85625 4 3000 A 0 0 0 0 B 0 110 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 feat_size
S 12067 1 3 1 0 6 1 12062 85635 4 3000 A 0 0 0 0 B 0 110 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 batch_sz
S 12068 14 5 0 0 0 1 12062 85288 0 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 5859 5 0 0 0 0 0 0 0 0 0 0 0 0 110 0 624 0 0 0 0 streaming_init streaming_init 
F 12068 5 12063 12064 12065 12066 12067
S 12069 23 5 0 0 0 12070 624 85303 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 streaming_start_epoch
S 12070 14 5 0 0 0 1 12069 85303 0 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 5865 0 0 0 0 0 0 0 0 0 0 0 0 0 193 0 624 0 0 0 0 streaming_start_epoch streaming_start_epoch 
F 12070 0
S 12071 23 5 0 0 0 12075 624 85325 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 streaming_get_batch
S 12072 7 3 2 0 4164 1 12071 85644 20000004 10003000 A 0 0 0 0 B 1 281 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 batch_data
S 12073 7 3 2 0 4167 1 12071 85655 20000004 10003000 A 0 0 0 0 B 1 281 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 batch_labels
S 12074 1 3 2 0 6 1 12071 85668 4 3000 A 0 0 0 0 B 0 281 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 actual_batch_size
S 12075 14 5 0 0 0 1 12071 85325 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 5866 3 0 0 0 0 0 0 0 0 0 0 0 0 281 0 624 0 0 0 0 streaming_get_batch streaming_get_batch 
F 12075 3 12072 12073 12074
S 12076 6 1 0 0 7 1 12071 85686 40800006 3000 A 0 0 0 0 B 0 282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_1_1
S 12077 6 1 0 0 7 1 12071 85694 40800006 3000 A 0 0 0 0 B 0 282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12078 6 1 0 0 7 1 12071 54824 40800006 3000 A 0 0 0 0 B 0 282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_4
S 12079 6 1 0 0 7 1 12071 54830 40800006 3000 A 0 0 0 0 B 0 282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5
S 12080 6 1 0 0 7 1 12071 54888 40800006 3000 A 0 0 0 0 B 0 282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6
S 12081 6 1 0 0 7 1 12071 85702 40800006 3000 A 0 0 0 0 B 0 282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_4455
S 12082 6 1 0 0 7 1 12071 85711 40800006 3000 A 0 0 0 0 B 0 282 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_4458
S 12083 6 1 0 0 7 1 12071 54900 40800006 3000 A 0 0 0 0 B 0 283 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_8
S 12084 6 1 0 0 7 1 12071 54906 40800006 3000 A 0 0 0 0 B 0 283 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_9
S 12085 6 1 0 0 7 1 12071 54949 40800006 3000 A 0 0 0 0 B 0 283 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_10
S 12086 6 1 0 0 7 1 12071 85720 40800006 3000 A 0 0 0 0 B 0 283 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_4465
S 12087 23 5 0 0 0 12090 624 85729 0 0 A 0 0 0 0 B 0 349 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 load_batch_sync
S 12088 1 3 3 0 4115 1 12087 55134 4 3000 A 0 0 0 0 B 0 349 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 buf
S 12089 1 3 1 0 6 1 12087 84715 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 batch_idx
S 12090 14 5 0 0 0 1 12087 85729 0 400000 A 0 0 0 0 B 0 349 0 0 0 0 0 5870 2 0 0 0 0 0 0 0 0 0 0 0 0 349 0 624 0 0 0 0 load_batch_sync load_batch_sync 
F 12090 2 12088 12089
S 12091 23 5 0 0 0 12094 624 85363 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 streaming_set_shuffle_mode
S 12092 1 3 1 0 6 1 12091 23043 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 mode
S 12093 1 3 1 0 6 1 12091 85745 80000004 3000 A 0 0 0 0 B 0 379 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 blk_size
S 12094 14 5 0 0 0 1 12091 85363 0 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 5873 2 0 0 0 0 0 0 0 0 0 0 0 0 379 0 624 0 0 0 0 streaming_set_shuffle_mode streaming_set_shuffle_mode 
F 12094 2 12092 12093
S 12095 23 5 0 0 9 12097 624 85390 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 streaming_get_num_batches
S 12096 1 3 0 0 6 1 12095 85754 4 1003000 A 0 0 0 0 B 0 393 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 num
S 12097 14 5 0 0 6 1 12095 85390 4 1400000 A 0 0 0 0 B 0 0 0 0 0 0 0 5876 0 0 0 12096 0 0 0 0 0 0 0 0 0 393 0 624 0 0 0 0 streaming_get_num_batches streaming_get_num_batches num
F 12097 0
S 12098 23 5 0 0 9 12100 624 85416 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 streaming_is_initialized
S 12099 1 3 0 0 18 1 12098 85758 4 1003000 A 0 0 0 0 B 0 401 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 initialized
S 12100 14 5 0 0 18 1 12098 85416 4 1400000 A 0 0 0 0 B 0 0 0 0 0 0 0 5877 0 0 0 12099 0 0 0 0 0 0 0 0 0 401 0 624 0 0 0 0 streaming_is_initialized streaming_is_initialized initialized
F 12100 0
S 12101 23 5 0 0 0 12102 624 85345 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 streaming_cleanup
S 12102 14 5 0 0 0 1 12101 85345 0 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 5878 0 0 0 0 0 0 0 0 0 0 0 0 0 409 0 624 0 0 0 0 streaming_cleanup streaming_cleanup 
F 12102 0
A 13 2 0 0 0 6 629 0 0 0 13 0 0 0 0 0 0 0 0 0 0 0
A 15 2 0 0 0 6 630 0 0 0 15 0 0 0 0 0 0 0 0 0 0 0
A 17 2 0 0 0 6 631 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0
A 68 1 0 0 0 58 684 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 71 1 0 0 0 67 686 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 78 2 0 0 0 6 715 0 0 0 78 0 0 0 0 0 0 0 0 0 0 0
A 80 2 0 0 0 6 716 0 0 0 80 0 0 0 0 0 0 0 0 0 0 0
A 82 2 0 0 0 6 717 0 0 0 82 0 0 0 0 0 0 0 0 0 0 0
A 87 2 0 0 0 6 718 0 0 0 87 0 0 0 0 0 0 0 0 0 0 0
A 89 2 0 0 0 6 719 0 0 0 89 0 0 0 0 0 0 0 0 0 0 0
A 91 2 0 0 0 6 720 0 0 0 91 0 0 0 0 0 0 0 0 0 0 0
A 93 2 0 0 0 6 721 0 0 0 93 0 0 0 0 0 0 0 0 0 0 0
A 95 2 0 0 0 6 722 0 0 0 95 0 0 0 0 0 0 0 0 0 0 0
A 97 2 0 0 0 6 723 0 0 0 97 0 0 0 0 0 0 0 0 0 0 0
A 99 2 0 0 0 6 724 0 0 0 99 0 0 0 0 0 0 0 0 0 0 0
A 102 2 0 0 0 6 725 0 0 0 102 0 0 0 0 0 0 0 0 0 0 0
A 104 2 0 0 0 6 726 0 0 0 104 0 0 0 0 0 0 0 0 0 0 0
A 106 2 0 0 0 6 727 0 0 0 106 0 0 0 0 0 0 0 0 0 0 0
A 108 2 0 0 0 6 728 0 0 0 108 0 0 0 0 0 0 0 0 0 0 0
A 110 2 0 0 0 6 729 0 0 0 110 0 0 0 0 0 0 0 0 0 0 0
A 112 2 0 0 0 6 730 0 0 0 112 0 0 0 0 0 0 0 0 0 0 0
A 114 2 0 0 0 6 731 0 0 0 114 0 0 0 0 0 0 0 0 0 0 0
A 116 2 0 0 0 6 732 0 0 0 116 0 0 0 0 0 0 0 0 0 0 0
A 118 2 0 0 0 6 733 0 0 0 118 0 0 0 0 0 0 0 0 0 0 0
A 120 2 0 0 0 6 734 0 0 0 120 0 0 0 0 0 0 0 0 0 0 0
A 122 2 0 0 0 6 735 0 0 0 122 0 0 0 0 0 0 0 0 0 0 0
A 124 2 0 0 0 6 736 0 0 0 124 0 0 0 0 0 0 0 0 0 0 0
A 126 2 0 0 0 6 737 0 0 0 126 0 0 0 0 0 0 0 0 0 0 0
A 141 1 0 0 0 97 750 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 142 2 0 0 128 6 935 0 0 0 142 0 0 0 0 0 0 0 0 0 0 0
A 143 2 0 0 0 6 938 0 0 0 143 0 0 0 0 0 0 0 0 0 0 0
A 144 2 0 0 0 6 848 0 0 0 144 0 0 0 0 0 0 0 0 0 0 0
A 147 2 0 0 0 6 944 0 0 0 147 0 0 0 0 0 0 0 0 0 0 0
A 148 2 0 0 0 6 939 0 0 0 148 0 0 0 0 0 0 0 0 0 0 0
A 175 2 0 0 0 6 849 0 0 0 175 0 0 0 0 0 0 0 0 0 0 0
A 177 2 0 0 0 6 850 0 0 0 177 0 0 0 0 0 0 0 0 0 0 0
A 179 2 0 0 0 6 851 0 0 0 179 0 0 0 0 0 0 0 0 0 0 0
A 181 2 0 0 0 6 852 0 0 0 181 0 0 0 0 0 0 0 0 0 0 0
A 364 2 0 0 127 6 934 0 0 0 364 0 0 0 0 0 0 0 0 0 0 0
A 438 2 0 0 129 6 936 0 0 0 438 0 0 0 0 0 0 0 0 0 0 0
A 442 2 0 0 0 6 937 0 0 0 442 0 0 0 0 0 0 0 0 0 0 0
A 448 2 0 0 0 6 940 0 0 0 448 0 0 0 0 0 0 0 0 0 0 0
A 450 2 0 0 0 6 941 0 0 0 450 0 0 0 0 0 0 0 0 0 0 0
A 452 2 0 0 0 6 942 0 0 0 452 0 0 0 0 0 0 0 0 0 0 0
A 454 2 0 0 0 6 943 0 0 0 454 0 0 0 0 0 0 0 0 0 0 0
A 465 2 0 0 0 6 947 0 0 0 465 0 0 0 0 0 0 0 0 0 0 0
A 490 2 0 0 0 6 959 0 0 0 490 0 0 0 0 0 0 0 0 0 0 0
A 594 2 0 0 0 6 1007 0 0 0 594 0 0 0 0 0 0 0 0 0 0 0
A 597 2 0 0 0 6 1008 0 0 0 597 0 0 0 0 0 0 0 0 0 0 0
A 673 2 0 0 0 6 1026 0 0 0 673 0 0 0 0 0 0 0 0 0 0 0
A 701 2 0 0 0 6 1027 0 0 0 701 0 0 0 0 0 0 0 0 0 0 0
A 705 2 0 0 0 7 1032 0 0 0 705 0 0 0 0 0 0 0 0 0 0 0
A 706 2 0 0 0 7 1033 0 0 0 706 0 0 0 0 0 0 0 0 0 0 0
A 707 2 0 0 0 7 1034 0 0 0 707 0 0 0 0 0 0 0 0 0 0 0
A 716 2 0 0 0 7 1044 0 0 0 716 0 0 0 0 0 0 0 0 0 0 0
A 718 2 0 0 0 7 1045 0 0 0 718 0 0 0 0 0 0 0 0 0 0 0
A 723 2 0 0 0 7 1046 0 0 0 723 0 0 0 0 0 0 0 0 0 0 0
A 1261 2 0 0 0 7 1049 0 0 0 1261 0 0 0 0 0 0 0 0 0 0 0
A 1263 2 0 0 0 7 1050 0 0 0 1263 0 0 0 0 0 0 0 0 0 0 0
A 1267 2 0 0 0 7 1051 0 0 0 1267 0 0 0 0 0 0 0 0 0 0 0
A 1452 2 0 0 0 7 1056 0 0 0 1452 0 0 0 0 0 0 0 0 0 0 0
A 1702 2 0 0 1199 6 7753 0 0 0 1702 0 0 0 0 0 0 0 0 0 0 0
A 1706 2 0 0 1352 6 7754 0 0 0 1706 0 0 0 0 0 0 0 0 0 0 0
A 1710 2 0 0 0 6 7755 0 0 0 1710 0 0 0 0 0 0 0 0 0 0 0
A 1960 2 0 0 0 6 7756 0 0 0 1960 0 0 0 0 0 0 0 0 0 0 0
A 1964 2 0 0 0 6 7757 0 0 0 1964 0 0 0 0 0 0 0 0 0 0 0
A 1968 2 0 0 0 6 7758 0 0 0 1968 0 0 0 0 0 0 0 0 0 0 0
A 1987 2 0 0 1357 6 7759 0 0 0 1987 0 0 0 0 0 0 0 0 0 0 0
A 2055 1 0 0 0 1493 7770 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2058 1 0 0 0 1493 7772 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2061 1 0 0 1073 1493 7774 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2064 1 0 0 0 1493 7776 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2067 1 0 0 0 1493 7778 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2070 1 0 0 0 1502 7783 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2073 1 0 0 0 1502 7785 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2076 1 0 0 0 1502 7787 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2079 1 0 0 0 1511 7792 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2082 1 0 0 0 1511 7794 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2085 1 0 0 0 1511 7796 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2088 1 0 0 720 1511 7798 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2091 1 0 0 0 1511 7800 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2094 1 0 0 0 1511 7802 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2097 1 0 0 0 1511 7804 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2100 1 0 0 0 1511 7806 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2103 1 0 0 0 1511 7808 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2106 1 0 0 0 1511 7810 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2109 1 0 0 0 1511 7812 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2112 1 0 0 0 1511 7814 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2115 1 0 0 802 1511 7816 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2118 1 0 0 0 1511 7818 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2121 1 0 0 0 1511 7820 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2124 1 0 0 0 1511 7822 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2127 1 0 0 0 1511 7824 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2130 1 0 0 0 1511 7826 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2133 1 0 0 0 1511 7828 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2136 1 0 0 0 1511 7830 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2139 1 0 0 0 1511 7832 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2142 1 0 0 0 1511 7834 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2145 1 0 0 0 1511 7836 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2148 1 0 0 0 1511 7838 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2151 1 0 0 0 1511 7840 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2154 1 0 0 1687 1511 7842 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2157 1 0 0 0 1511 7844 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2160 1 0 0 0 1511 7846 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2163 1 0 0 0 1511 7848 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2166 1 0 0 0 1511 7850 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2169 1 0 0 2029 1511 7852 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2172 1 0 0 0 1511 7854 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2175 1 0 0 0 1511 7856 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2178 1 0 0 0 1520 7861 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2181 1 0 0 0 1520 7863 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2184 1 0 0 0 1520 7865 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2187 1 0 0 0 1520 7867 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2190 1 0 0 0 1520 7869 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2193 1 0 0 0 1520 7871 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2196 1 0 0 0 1520 7873 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2199 1 0 0 0 1520 7875 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2202 1 0 0 0 1520 7877 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2205 1 0 0 0 1520 7879 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2208 1 0 0 0 1520 7881 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2211 1 0 0 0 1520 7883 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2214 1 0 0 0 1520 7885 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2217 1 0 0 0 1520 7887 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2220 1 0 0 0 1520 7889 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2223 1 0 0 0 1529 7894 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2226 1 0 0 0 1529 7896 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2229 1 0 0 0 1529 7898 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2232 1 0 0 0 1529 7900 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2235 1 0 0 0 1529 7902 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2238 1 0 0 0 1529 7904 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2241 1 0 0 0 1529 7906 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2244 1 0 0 0 1529 7908 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2247 1 0 0 0 1529 7910 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2250 1 0 0 1026 1529 7912 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2253 1 0 0 0 1529 7914 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2256 1 0 0 0 1529 7916 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2259 1 0 0 0 1529 7918 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2262 1 0 0 0 1529 7920 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2265 1 0 0 0 1529 7922 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2268 1 0 0 0 1529 7924 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2271 1 0 0 0 1529 7926 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2274 1 0 0 0 1529 7928 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2277 1 0 0 0 1529 7930 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2280 1 0 0 0 1529 7932 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2283 1 0 0 0 1529 7934 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2286 1 0 0 0 1529 7936 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2289 1 0 0 1053 1529 7938 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2292 1 0 0 0 1529 7940 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2295 1 0 0 0 1529 7942 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2298 1 0 0 0 1529 7944 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2301 1 0 0 0 1529 7946 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2304 1 0 0 0 1529 7948 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2307 1 0 0 0 1538 7953 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2310 1 0 0 0 1538 7955 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2313 1 0 0 712 1538 7957 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2316 1 0 0 0 1538 7959 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2319 1 0 0 0 1538 7961 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2322 1 0 0 0 1538 7963 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2325 1 0 0 0 1538 7965 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2328 1 0 0 0 1538 7967 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2331 1 0 0 0 1538 7969 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2334 1 0 0 0 1538 7971 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2337 1 0 0 0 1547 7976 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2340 1 0 0 0 1547 7978 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2343 1 0 0 0 1547 7980 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2346 1 0 0 0 1547 7982 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2349 1 0 0 0 1547 7984 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2352 1 0 0 0 1547 7986 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2355 1 0 0 0 1547 7988 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2358 1 0 0 2148 1556 7993 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2361 1 0 0 2151 1556 7995 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2364 1 0 0 2154 1556 7997 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2367 1 0 0 2157 1556 7999 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2370 1 0 0 2160 1556 8001 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2373 1 0 0 2163 1556 8003 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2376 1 0 0 1821 1565 8008 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2379 1 0 0 0 1574 8013 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2382 1 0 0 0 1574 8015 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2385 1 0 0 0 1583 8020 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2388 1 0 0 0 1583 8022 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2391 1 0 0 0 1592 8027 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2394 1 0 0 0 1592 8029 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4403 1 0 7 0 4124 11999 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4404 10 0 0 1142 7 4403 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 716
A 4405 10 0 0 4404 7 4403 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 718
A 4406 4 0 0 3808 7 4405 0 11 0 0 0 0 2 0 0 0 0 0 0 0 0
A 4407 4 0 0 0 7 4404 0 4406 0 0 0 0 1 0 0 0 0 0 0 0 0
A 4408 10 0 0 4405 7 4403 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 1261
A 4409 10 0 0 4408 7 4403 19 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 1263
A 4410 4 0 0 3791 7 4409 0 11 0 0 0 0 2 0 0 0 0 0 0 0 0
A 4411 4 0 0 0 7 4408 0 4410 0 0 0 0 1 0 0 0 0 0 0 0 0
A 4412 10 0 0 4409 7 4403 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 707
A 4413 10 0 0 4412 7 4403 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 1267
A 4414 10 0 0 4413 7 4403 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 723
A 4415 10 0 0 4414 7 4403 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 705
A 4417 1 0 9 0 4130 12005 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4418 10 0 0 0 7 4417 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 716
A 4419 10 0 0 4418 7 4417 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 718
A 4420 4 0 0 3818 7 4419 0 11 0 0 0 0 2 0 0 0 0 0 0 0 0
A 4421 4 0 0 4209 7 4418 0 4420 0 0 0 0 1 0 0 0 0 0 0 0 0
A 4422 10 0 0 4419 7 4417 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 707
A 4423 10 0 0 4422 7 4417 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 723
A 4424 10 0 0 4423 7 4417 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 705
A 4427 2 0 0 1152 7 12016 0 0 0 4427 0 0 0 0 0 0 0 0 0 0 0
A 4434 1 0 1 0 4157 12040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4435 10 0 0 0 7 4434 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 716
A 4436 10 0 0 4435 7 4434 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 718
A 4437 4 0 0 3828 7 4436 0 11 0 0 0 0 2 0 0 0 0 0 0 0 0
A 4438 4 0 0 0 7 4435 0 4437 0 0 0 0 1 0 0 0 0 0 0 0 0
A 4439 10 0 0 4436 7 4434 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 707
A 4440 10 0 0 4439 7 4434 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 723
A 4441 10 0 0 4440 7 4434 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
X 1 705
A 4447 1 0 0 0 7 12080 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4448 1 0 0 1224 7 12076 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4449 1 0 0 1293 7 12081 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4450 1 0 0 1422 7 12078 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4451 1 0 0 1221 7 12077 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4452 1 0 0 0 7 12082 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4453 1 0 0 0 7 12079 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4454 1 0 0 0 7 12085 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4455 1 0 0 1225 7 12083 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4456 1 0 0 1430 7 12086 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 4457 1 0 0 0 7 12084 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Z
J 133 1 1
V 68 58 7 0
S 0 58 0 0 0
A 0 6 0 0 1 2 0
J 134 1 1
V 71 67 7 0
S 0 67 0 0 0
A 0 6 0 0 1 2 0
J 53 1 1
V 141 97 7 0
S 0 97 0 0 0
A 0 76 0 0 1 68 0
J 21 1 1
V 2055 1493 7 0
S 0 1493 0 0 0
A 0 6 0 0 1 1702 0
J 21 1 1
V 2058 1493 7 0
S 0 1493 0 0 0
A 0 6 0 0 1 1706 0
J 21 1 1
V 2061 1493 7 0
S 0 1493 0 0 0
A 0 6 0 0 1 1710 0
J 21 1 1
V 2064 1493 7 0
S 0 1493 0 0 0
A 0 6 0 0 1 673 0
J 21 1 1
V 2067 1493 7 0
S 0 1493 0 0 0
A 0 6 0 0 1 364 0
J 32 1 1
V 2070 1502 7 0
S 0 1502 0 0 0
A 0 6 0 0 1 3 0
J 32 1 1
V 2073 1502 7 0
S 0 1502 0 0 0
A 0 6 0 0 1 15 0
J 32 1 1
V 2076 1502 7 0
S 0 1502 0 0 0
A 0 6 0 0 1 97 0
J 41 1 1
V 2079 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 3 0
J 41 1 1
V 2082 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 15 0
J 41 1 1
V 2085 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 17 0
J 41 1 1
V 2088 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 102 0
J 41 1 1
V 2091 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 112 0
J 41 1 1
V 2094 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 114 0
J 41 1 1
V 2097 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 116 0
J 41 1 1
V 2100 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 108 0
J 41 1 1
V 2103 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 110 0
J 41 1 1
V 2106 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 120 0
J 41 1 1
V 2109 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 122 0
J 41 1 1
V 2112 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 124 0
J 41 1 1
V 2115 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 126 0
J 41 1 1
V 2118 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 144 0
J 41 1 1
V 2121 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 142 0
J 41 1 1
V 2124 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 438 0
J 41 1 1
V 2127 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 175 0
J 41 1 1
V 2130 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 177 0
J 41 1 1
V 2133 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 442 0
J 41 1 1
V 2136 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 143 0
J 41 1 1
V 2139 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 179 0
J 41 1 1
V 2142 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 148 0
J 41 1 1
V 2145 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 181 0
J 41 1 1
V 2148 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 448 0
J 41 1 1
V 2151 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 450 0
J 41 1 1
V 2154 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 452 0
J 41 1 1
V 2157 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 454 0
J 41 1 1
V 2160 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 147 0
J 41 1 1
V 2163 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 97 0
J 41 1 1
V 2166 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 87 0
J 41 1 1
V 2169 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 78 0
J 41 1 1
V 2172 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 99 0
J 41 1 1
V 2175 1511 7 0
S 0 1511 0 0 0
A 0 6 0 0 1 594 0
J 82 1 1
V 2178 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 2 0
J 82 1 1
V 2181 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 3 0
J 82 1 1
V 2184 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 97 0
J 82 1 1
V 2187 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 99 0
J 82 1 1
V 2190 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 17 0
J 82 1 1
V 2193 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 114 0
J 82 1 1
V 2196 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 118 0
J 82 1 1
V 2199 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 80 0
J 82 1 1
V 2202 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 82 0
J 82 1 1
V 2205 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 89 0
J 82 1 1
V 2208 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 91 0
J 82 1 1
V 2211 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 93 0
J 82 1 1
V 2214 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 95 0
J 82 1 1
V 2217 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 104 0
J 82 1 1
V 2220 1520 7 0
S 0 1520 0 0 0
A 0 6 0 0 1 106 0
J 123 1 1
V 2223 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 15 0
J 123 1 1
V 2226 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 78 0
J 123 1 1
V 2229 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 80 0
J 123 1 1
V 2232 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 82 0
J 123 1 1
V 2235 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 2 0
J 123 1 1
V 2238 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 13 0
J 123 1 1
V 2241 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 3 0
J 123 1 1
V 2244 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 87 0
J 123 1 1
V 2247 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 89 0
J 123 1 1
V 2250 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 91 0
J 123 1 1
V 2253 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 93 0
J 123 1 1
V 2256 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 95 0
J 123 1 1
V 2259 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 97 0
J 123 1 1
V 2262 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 99 0
J 123 1 1
V 2265 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 17 0
J 123 1 1
V 2268 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 102 0
J 123 1 1
V 2271 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 104 0
J 123 1 1
V 2274 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 106 0
J 123 1 1
V 2277 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 108 0
J 123 1 1
V 2280 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 110 0
J 123 1 1
V 2283 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 112 0
J 123 1 1
V 2286 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 114 0
J 123 1 1
V 2289 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 116 0
J 123 1 1
V 2292 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 118 0
J 123 1 1
V 2295 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 120 0
J 123 1 1
V 2298 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 122 0
J 123 1 1
V 2301 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 124 0
J 123 1 1
V 2304 1529 7 0
S 0 1529 0 0 0
A 0 6 0 0 1 126 0
J 157 1 1
V 2307 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 3 0
J 157 1 1
V 2310 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 1960 0
J 157 1 1
V 2313 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 1964 0
J 157 1 1
V 2316 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 1968 0
J 157 1 1
V 2319 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 13 0
J 157 1 1
V 2322 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 89 0
J 157 1 1
V 2325 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 490 0
J 157 1 1
V 2328 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 701 0
J 157 1 1
V 2331 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 597 0
J 157 1 1
V 2334 1538 7 0
S 0 1538 0 0 0
A 0 6 0 0 1 1987 0
J 173 1 1
V 2337 1547 7 0
S 0 1547 0 0 0
A 0 6 0 0 1 2 0
J 173 1 1
V 2340 1547 7 0
S 0 1547 0 0 0
A 0 6 0 0 1 3 0
J 173 1 1
V 2343 1547 7 0
S 0 1547 0 0 0
A 0 6 0 0 1 15 0
J 173 1 1
V 2346 1547 7 0
S 0 1547 0 0 0
A 0 6 0 0 1 97 0
J 173 1 1
V 2349 1547 7 0
S 0 1547 0 0 0
A 0 6 0 0 1 13 0
J 173 1 1
V 2352 1547 7 0
S 0 1547 0 0 0
A 0 6 0 0 1 87 0
J 173 1 1
V 2355 1547 7 0
S 0 1547 0 0 0
A 0 6 0 0 1 78 0
J 193 1 1
V 2358 1556 7 0
S 0 1556 0 0 0
A 0 6 0 0 1 2 0
J 193 1 1
V 2361 1556 7 0
S 0 1556 0 0 0
A 0 6 0 0 1 3 0
J 193 1 1
V 2364 1556 7 0
S 0 1556 0 0 0
A 0 6 0 0 1 15 0
J 193 1 1
V 2367 1556 7 0
S 0 1556 0 0 0
A 0 6 0 0 1 97 0
J 193 1 1
V 2370 1556 7 0
S 0 1556 0 0 0
A 0 6 0 0 1 13 0
J 193 1 1
V 2373 1556 7 0
S 0 1556 0 0 0
A 0 6 0 0 1 87 0
J 211 1 1
V 2376 1565 7 0
S 0 1565 0 0 0
A 0 6 0 0 1 2 0
J 219 1 1
V 2379 1574 7 0
S 0 1574 0 0 0
A 0 6 0 0 1 2 0
J 219 1 1
V 2382 1574 7 0
S 0 1574 0 0 0
A 0 6 0 0 1 3 0
J 227 1 1
V 2385 1583 7 0
S 0 1583 0 0 0
A 0 6 0 0 1 2 0
J 227 1 1
V 2388 1583 7 0
S 0 1583 0 0 0
A 0 6 0 0 1 3 0
J 235 1 1
V 2391 1592 7 0
S 0 1592 0 0 0
A 0 6 0 0 1 2 0
J 235 1 1
V 2394 1592 7 0
S 0 1592 0 0 0
A 0 6 0 0 1 3 0
T 1702 378 0 3 0 0
R 1707 384 0 0
A 0 7 0 705 1 10 0
T 11995 4115 0 0 0 0
A 12000 7 4133 0 1 2 1
A 11999 7 0 1452 1 10 1
A 12006 7 4135 0 1 2 1
A 12005 7 0 1261 1 10 0
Z
