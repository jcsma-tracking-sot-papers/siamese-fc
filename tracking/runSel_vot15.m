function runSel_vot15
    
    clear all
    close all
    clc
    visualization = true;
    gpus = 0;

    seq={'vot15_bag'};
    seq={'vot15_basketball'};
    seq={'vot15_motocross1'};

    for s=1:numel(seq)
       run_tracker(seq{s}, visualization);
    end
end
