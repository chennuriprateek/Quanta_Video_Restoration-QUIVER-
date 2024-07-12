def emvd_training_args(parser):
    # dataset
    parser.add_argument('--gtdata_dir', type=str, default='', help='clean videoframes directory ')
    parser.add_argument('--valgtdata_dir', type=str, default='', help='clean val videoframes directory ')
    parser.add_argument('--plotdir', type=str, default='', help='clean val videoframes directory ')
    parser.add_argument('--gray_mode', type=bool, default=True, help='image transforms')
    parser.add_argument('--downsample', default=None, help='downsample ratio for the input to the model')

    parser.add_argument('--image_type', type=str, default='*.png', help='*.png or *.tiff etc.')
    parser.add_argument('--num_frames', type=int, default=11, help='max frames that should be loaded at once. '
                                                                   'Similar to batch size for spatial model training')                                           
    parser.add_argument('--transforms', type=bool, default=True, help='image transforms')
    parser.add_argument('--patch_size', type=int, default=128, help='patch size of each frame during training')

    # arch
    parser.add_argument('--future_frames', type=int, default=5, help='...')
    parser.add_argument('--past_frames', type=int, default=5, help='...')
    
    # training
    parser.add_argument('--loss_fun_name', type=str, default='L1', help='cost function')
    parser.add_argument('--model_name', type=str, default='emvd', help='model name')
    parser.add_argument('--weights_dir', type=str, default='', help='weights master directory')
    parser.add_argument('--load_model_flag', type=bool, default=False, help='load previously stored model weights')
    parser.add_argument('--start_over', type=bool, default=False, help='load previously stored model weights')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=0, help='....')
    parser.add_argument('-batch_size', type=int, default=4, help='....')
    parser.add_argument('-save_period', type=int, default=300, help='....')
    parser.add_argument('-total_epochs', type=int, default=1000, help='....')
    parser.add_argument('-start_epoch', type=int, default=0, help='....')

    parser.add_argument('-log_every', type=int, default=1, help='....')
    parser.add_argument('--visualize', type=bool, default=True, help='visualization during validation')


def emvd_testing_args(parser):
    parser.add_argument('--testgtdata_dir', type=str, default='', help='clean val videoframes directory ')
    parser.add_argument('--plotdir', type=str, default='', help='...')
    parser.add_argument('--folder_name', type=str, default='', help='folder name')
    parser.add_argument('--gray_mode', type=bool, default=True, help='image transforms')
    parser.add_argument('--downsample', default=None, help='downsample ratio for the input to the model')

    parser.add_argument('--image_type', type=str, default='*.png', help='*.png or *.tiff etc.')
    parser.add_argument('--num_frames', type=int, default=11, help='max frames that should be loaded at once. '
                                                                   'Similar to batch size for spatial model training')

    # arch
    parser.add_argument('--future_frames', type=int, default=5, help='...')
    parser.add_argument('--past_frames', type=int, default=5, help='...')
    

    # testing
    parser.add_argument('--model_name', type=str, default='emvd', help='model name')
    parser.add_argument('-batch_size', type=int, default=1, help='....')
    parser.add_argument('--weights_path', type=str, default='', help='weights master directory')
    parser.add_argument('--save_path', type=str, default='', help='...')
    parser.add_argument('--visualize', type=bool, default=False, help='visualization during validation')


def sensor_args(parser):
    parser.add_argument('-FWC', type=int, default=200, help='....')
    parser.add_argument('-avg_PPP', type=float, default=3.25, help='....')
    parser.add_argument('-gain', type=float, default=0.5 * (7/3.25), help='....')
    parser.add_argument('-Nbits', type=int, default=3, help='....')
    parser.add_argument('-QE', type=float, default=0.8, help='....')
    parser.add_argument('-theta_dark', type=float, default=1.6, help='....')
    parser.add_argument('-sigma_read', type=float, default=0.2, help='....')
    parser.add_argument('-clicks_per_frame', type=int, default=1, help='....')
