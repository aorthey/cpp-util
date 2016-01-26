#!/usr/bin/perl

open FILE, "results.tmp" or die $!;

my ($data, $n, $offset); 

print "Features Hypotheses ModelCUDAtime ModelCPUtime CompleteTimeCUDA CompleteTimeCPU\n";
while (($n = read FILE, $data, 10000, $offset) != 0) { 
	$match='\[CUDA\] Computing model : ([0-9]+(?:\.[0-9]+)?)
\[CPU\] Computing Model: ([0-9]+(?:\.[0-9]+)?)
Features: ([0-9]+)
Number of Hypotheses: ([0-9]+)
CPU: ([0-9]+(?:\.[0-9]+)?)
CUDA: ([0-9]+(?:\.[0-9]+)?)';
	while($data =~ m/$match/g){
		print "$3 $4 $1 $2 $6 $5\n";
	}

}
close FILE;


