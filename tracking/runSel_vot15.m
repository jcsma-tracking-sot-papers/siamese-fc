function runSel_vot15
    
    clear all
    close all
    clc
    visualization = true;
    gpus = 0;

    seq={
    'vot15_bag'    
    };

    for s=1:numel(seq)
       run_tracker(seq{s}, visualization);
    end
end
