def distill_args(parser):
    """Extra CLI arguments for knowledge distillation.

    Usage (append after quiver_training_args and sensor_args):
        quiver_qis_distill_args.distill_args(parser)
    """
    # Teacher
    parser.add_argument('--teacher_weights', type=str, default='',
                        help='Path to pre-trained teacher checkpoint (.pth)')
    parser.add_argument('--teacher_n_features', type=int, default=64,
                        help='n_features of the teacher model')
    parser.add_argument('--teacher_n_blocks', type=int, default=12,
                        help='n_blocks of the teacher model')

    # Student (inherits n_features / n_blocks from quiver_training_args)
    parser.add_argument('--student_weights', type=str, default='',
                        help='Optional path to resume student checkpoint')

    # Distillation loss weights
    parser.add_argument('--lambda_kd_hf3', type=float, default=1.0,
                        help='Weight for RDBCell bottleneck (hf3) feature loss')
    parser.add_argument('--lambda_kd_att', type=float, default=0.5,
                        help='Weight for spatial_att output feature loss')
    parser.add_argument('--lambda_kd_hidden', type=float, default=0.5,
                        help='Weight for recurrent hidden state (s) feature loss')
    parser.add_argument('--lambda_kd_warp', type=float, default=0.25,
                        help='Weight for aligned/fused feature loss (x_warped)')

    # Task loss weight relative to distillation
    parser.add_argument('--lambda_task', type=float, default=1.0,
                        help='Weight applied to the standard task loss')
