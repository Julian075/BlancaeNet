%%% LRR algorithms
% struct = run_algorithm_lrr(string, 2dmatrix)
%
function results = run_algorithm_2(method_id, algorithm_id, data, params)
results = struct('cputime', 0, 'L', [], 'S', [], 'O', []);
%try  
%lrs_load_conf;
%catch
%errordlg("Load error Lrs conf ")
%end
  disp('Running RPCA');
  alg_path=[pwd '/algorithms/rpca/NSA1/'];
  actual_path=pwd;
  cd(alg_path);
  M = data; T = data;
  L = zeros(size(data)); % low-rank component
  S = zeros(size(data)); % sparse component
  results.cputime = 0;
  try
  if(isempty(params))
    params.rows = size(data,1);
    params.cols = size(data,2);
  end
  catch ME
            errordlg(ME.identifier);
            errordlg('Error params');
  end
  timerVal = tic;
  % warning('off','all');
  try
  [L,S]=run_alg_2(M);
  % warning('on','all');
  cputime = toc(timerVal);
  catch ME
            errordlg(ME.identifier);
            errordlg('Error run alg');
  end
  try  
  results.L = L; % low-rank component
  results.S = S; % sparse component
  results.O = hard_threshold(S); % apply hard thresholding
  results.cputime = cputime;
	disp('Decomposition finished');
   cd(actual_path);
    catch ME
            errordlg(ME);
            errordlg('Error NSA Part 3');
  end
end
