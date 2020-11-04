% -------------------------------------------------------------------------------------------------------------------------
function [newTargetPosition, bestScale,responseMap_w_best, x_crop_best] = tracker_eval(net_x, s_x, scoreId, z_features, x_crops, targetPosition, window, p)
%TRACKER_STEP
%   runs a forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
%   reusing the features of the exemplar z computed at the first frame.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------

%% forward pass, using the pyramid of scaled crops as a "batch"
%code for the correlation is performed in the file 'XCorr'
net_x.eval({p.id_feat_z, z_features, 'instance', x_crops});

%reshape response Maps according to the test scales
responseMaps = reshape(net_x.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);

if p.debug==true
    figure(2000);
    for i=1:size(responseMaps,3)
        subplot(2,size(responseMaps,3),i)
        imshow(uint8(x_crops(:,:,:,i))); 
        title(sprintf('%dx%dx%d',size(x_crops,1),size(x_crops,2),size(x_crops,3)))
                
        subplot(2,size(responseMaps,3),i+size(responseMaps,3))
        imagesc(responseMaps(:,:,i));
        colorbar;
        title(sprintf('Map #%d (%dx%d)',i,size(responseMaps,1),size(responseMaps,2)));
    end
end

%% Choose the scale whose response map has the highest peak
if isempty(p.gpus)
    responseMapsUP = single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale));
else
    responseMapsUP = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale)));
end

if p.numScale>1
    currentScaleID = ceil(p.numScale/2);
    bestScale = currentScaleID;
    bestPeak = -Inf;
    
    for s=1:p.numScale
        
        %upsample scale if needed
        if p.responseUp > 1
            % upsample to improve accuracy
            responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp, 'bicubic');
        else
            responseMapsUP(:,:,s) = responseMaps(:,:,s);
        end
        thisResponse = responseMapsUP(:,:,s);
        
        % penalize change of scale (NOT CLEAR WHY THIS IS DONE!)
        if s~=currentScaleID
            thisResponse = thisResponse * p.scalePenalty; 
        end
        thisPeak = max(thisResponse(:));
        
        %keep the scale if the peak is the new max value
        if thisPeak > bestPeak
            bestPeak = thisPeak; 
            bestScale = s; 
        end
    end
    
    %get the selected scale map
    responseMap = responseMapsUP(:,:,bestScale);
else
    responseMap = responseMapsUP;
    bestScale = 1;
end

%% normalization and finding of target location
% normalize the response map
responseMap = responseMap - min(responseMap(:)); % make the response map from 0 to maxVal
responseMap = responseMap / sum(responseMap(:)); % make the response map sum to 1

% apply windowing
responseMap_w = (1-p.wInfluence)*responseMap + p.wInfluence*window;

%find the best location
[r_max, c_max] = find(responseMap_w == max(responseMap_w(:)), 1);
[r_max, c_max] = avoid_empty_position(r_max, c_max, p);
p_corr = [r_max, c_max];

if p.debug==true
    figure(2001);
    
    subplot 121; imagesc(responseMap); colorbar; title(sprintf('Responsed Map \nbestScale=%d',bestScale));
    subplot 122; imagesc(responseMap_w); colorbar; title(sprintf('Responsed Map*win'));
    hold on; plot(c_max,r_max,'r+');
end

%% Convert to crop-relative coordinates to frame coordinates
% displacement from the center in instance final representation ...
disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);
% ... in instance input ...
disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
% ... in instance original crop (in frame coordinates)
disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
% position within frame in frame coordinates
newTargetPosition = targetPosition + disp_instanceFrame;

responseMap_w_best = responseMap_w;
x_crop_best = x_crops(:,:,:,bestScale);

end

function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
if isempty(r_max)
    r_max = ceil(params.scoreSize/2);
end
if isempty(c_max)
    c_max = ceil(params.scoreSize/2);
end
end
