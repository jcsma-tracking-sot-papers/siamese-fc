function [im] = load_frame(imgFile,p)
% load new frame on GPU
if isempty(p.gpus)
    im = single(imgFile);
else
    im = gpuArray(single(imgFile));
end

% if grayscale repeat one channel to match filters size
if(size(im, 3)==1)
    im = repmat(im, [1 1 3]);
end

end

