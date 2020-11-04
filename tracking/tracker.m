function bboxes = tracker(varargin)
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
%% Settings
% These are the default hyper-params for SiamFC-3S
% The ones for SiamFC (5 scales) are in params-5s.txt
p.numScale = 3;
p.scaleStep = 1.0375;
p.scalePenalty = 0.9745;
p.scaleLR = 0.59; % damping factor for scale update
p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
p.windowing = 'cosine'; % to penalize large displacements
p.wInfluence = 0.176; % windowing influence (in convex sum)
p.net = '2016-08-17.net.mat';

% execution, visualization, benchmark
p.video = 'vot15_basketball';
p.visualization = true;
p.debug=false;
p.gpus = [];
p.bbox_output = false;
p.fout = -1;

% Params from the network architecture, have to be consistent with the training
p.exemplarSize = 127;  % input z size
p.instanceSize = 255;  % input x size (search region)
p.scoreSize = 17;
p.totalStride = 8;
p.contextAmount = 0.5; % context amount for the exemplar
p.subMean = false;

% SiamFC prefix and ids
p.prefix_z = 'a_'; % used to identify the layers of the exemplar
p.prefix_x = 'b_'; % used to identify the layers of the instance
p.prefix_join = 'xcorr';
p.prefix_adj = 'adjust';
p.id_feat_z = 'a_feat';
p.id_score = 'score';

% Overwrite default parameters with varargin
p = vl_argparse(p, varargin);

%% Paths
% Get environment-specific default paths.
p = env_paths_tracking(p);
% Load ImageNet Video statistics
if exist(p.stats_path,'file')
    stats = load(p.stats_path);
else
    warning('No stats found at %s', p.stats_path);
    stats = [];
end

%% Networks
% Load two copies of the pre-trained network
net_z = load_pretrained([p.net_base_path p.net], p.gpus); %copy one for target (exemplar)
net_x = load_pretrained([p.net_base_path p.net], []); %copy two for target+search area (instance)

% Get exemplar branch (used only once per video) computes features for the target
remove_layers_from_prefix(net_z, p.prefix_x);
remove_layers_from_prefix(net_z, p.prefix_join);
remove_layers_from_prefix(net_z, p.prefix_adj);

% Get instance branch computes features for search region x and cross-correlates with z features
remove_layers_from_prefix(net_x, p.prefix_z);

% Get Ids of last layer for both branches
zFeatId = net_z.getVarIndex(p.id_feat_z); %feature layer
scoreId = net_x.getVarIndex(p.id_score); %score map layer

%% Load video info and display settings
[imgFiles, targetPosition, targetSize] = load_video_info(p.seq_base_path, p.video);
nImgs = numel(imgFiles);
startFrame = 1;

% get the first frame of the video
im = load_frame(imgFiles{startFrame},p);

% Init visualization
videoPlayer = [];
%if p.visualization && isToolboxAvailable('Computer Vision Toolbox')
if p.visualization && isToolboxAvailable('Computer Vision Toolbox (diasbles)')
    videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
end

v = VideoWriter(sprintf('%s_result.mp4',p.video),'Uncompressed AVI');
%v.Quality = 95;
v.FrameRate = 8;
open(v);

% get avg for padding
avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

%% Target initialization
wc_z = targetSize(2) + p.contextAmount*sum(targetSize);
hc_z = targetSize(1) + p.contextAmount*sum(targetSize);
s_z = sqrt(wc_z*hc_z);

% initialize the exemplar
[z_crop_img, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);

if p.subMean
    z_crop = bsxfun(@minus, z_crop_img, reshape(stats.z.rgbMean, [1 1 3]));
else
    z_crop = z_crop_img;
end

% evaluate the offline-trained network for exemplar z features
net_z.eval({'exemplar', z_crop});

% get the features of the evaluated network
z_features = net_z.vars(zFeatId).value;

% replicate features for each of the scales being tested
z_features_scales = repmat(z_features, [1 1 1 p.numScale]);

if p.debug==true
    figure(1);
    subplot 121; imshow(uint8(z_crop_img));
    title(sprintf('target expanded \n%dx%dx%d',size(z_crop_img,1),size(z_crop_img,2),size(z_crop_img,3)))
    subplot 122; imshow(uint8(z_crop))
    title(sprintf('target expanded \n(Mean Avg)%dx%dx%d',size(z_crop_img,1),size(z_crop_img,2),size(z_crop_img,3)))
end

%% Scale settings
%scales being tested
scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));

%offset to define the search area around target according to network inputs (in pixels)
d_search = (p.instanceSize - p.exemplarSize)/2;

%offset to define the search area adapted to actual target size
scale_z = p.exemplarSize / s_z; %scale relation between network input and actual target
pad = d_search/scale_z; %offset according to actual target scale (in pixels)
s_x = s_z + 2*pad;
% arbitrary scale saturation
min_s_x = 0.2*s_x;
max_s_x = 5*s_x;

%% Spatial window settings
%Cosine window is added to the score map to penalize large displacements
switch p.windowing
    case 'cosine'
        window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
    case 'uniform'
        window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
end
% make the window sum 1
window = window / sum(window(:));

%% start tracking
bboxes = zeros(nImgs, 4);

tic;
for i = startFrame:nImgs
    if i>startFrame
        im = load_frame(imgFiles{i},p);
        
        %get scales for instance (search area)
        scaledInstance = s_x .* scales;
        
        %get scales for target (object to track)
        scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];
        
        % extract scaled crops for search region x at previous target position
        x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
        
        % evaluate the offline-trained network for exemplar x features
        [newTargetPosition, newScale,responseMap_w_best, x_crop_best] = tracker_eval(net_x, round(s_x), scoreId, z_features_scales, x_crops, targetPosition, window, p);
        
        targetPosition = gather(newTargetPosition);
        
        % scale damping and saturation
        s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
        targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
    else
        % at the first frame output position and size passed as input (ground truth)
    end
    
    rectPosition = [targetPosition([2,1]) - targetSize([2,1])/2, targetSize([2,1])];
    % output bbox in the original frame coordinates
    oTargetPosition = targetPosition; % .* frameSize ./ newFrameSize;
    oTargetSize = targetSize; % .* frameSize ./ newFrameSize;
    bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];
    
    if p.visualization
        if isempty(videoPlayer)
            %             figure(1000), imshow(im/255);
            %             figure(1000), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
            %             drawnow
            %             fprintf('Frame %d\n', startFrame+i);
            
            if i>startFrame
                %                 ha = tight_subplot(2,2,[.01 .03],[.1 .01],[.01 .01]);
                %
                %                 axes(ha(1)); axis off; imshow(im/255); hold on;
                %                 rectangle('Position', rectPosition, 'LineWidth', 2, 'EdgeColor', 'y');
                %                 text(10,20,sprintf('Frame %03d',i),'color','r','fontsize',12);
                %                 axes(ha(2)); axis off; imshow(uint8(z_crop_img));
                %                 text(5,5,'Target (Exemplar)','color','r','fontsize',12);
                %                 axes(ha(3)); axis off; imshow(uint8(x_crop_best));
                %                 text(10,20,sprintf('Search area %03d',i),'color','r','fontsize',12);
                %                 axes(ha(4)); axis off; imagesc(responseMap_w_best); colorbar;
                %                 text(10,20,sprintf('Response map %03d',i),'color','r','fontsize',12);
                %                 set(ha(1:4),'XTickLabel',''); set(ha,'YTickLabel','')
                
                figure(99); clf;
                subplot_tight(2, 2, 1, [0.0001]);
                imshow(im/255); hold on;
                rectangle('Position', rectPosition, 'LineWidth', 2, 'EdgeColor', 'y');
                text(10,20,sprintf('Frame %03d',i),'color','r','fontsize',12);
                subplot_tight(2, 2, 2, [0.0001]);
                imshow(uint8(z_crop_img));
                text(5,5,'Target (Exemplar)','color','r','fontsize',12);
                subplot_tight(2, 2, 3, [0.0001]);                
                imshow(uint8(x_crop_best));  axis off;
                text(10,20,sprintf('Search area %03d',i),'color','r','fontsize',12);
                subplot_tight(2, 2, 4, [0.0001]);
                imagesc(responseMap_w_best); colorbar; axis off;
                text(10,20,sprintf('Response map %03d',i),'color','r','fontsize',12);
                frame = getframe(gcf);
                writeVideo(v,frame);
            end
        else
            im = gather(im)/255;
            im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
            % Display the annotated video frame using the video player object.
            step(videoPlayer, im);
        end
    end
    
    if p.bbox_output
        fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', bboxes(i, :));
    end
    
end
close(v);
bboxes = bboxes(startFrame : i, :);

end
