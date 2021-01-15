use strict;

use File::Spec::Functions 'catfile';
use File::Spec::Functions 'catdir';
use List::MoreUtils qw(firstidx);
use List::Util qw(sum);
use List::MoreUtils 'pairwise';

my $svm_cache = {};

sub get_svm_info {
	my $svm_dir = shift;
	my %svm_cache;
	$svm_cache{svm} = read_file_into_2d_array(catfile($svm_dir, 'sv.txt'));
	$svm_cache{center} = read_file_into_hash(catfile($svm_dir, 'center.txt'));
	$svm_cache{scale} = read_file_into_hash(catfile($svm_dir, 'scale.txt'));
	$svm_cache{misc} = read_file_into_hash(catfile($svm_dir, 'misc.txt'));
	return \%svm_cache;
}


sub svm {
	my ($cacheref, $featref, $kernel) = @_[0..2];
	my %cache = % { $cacheref };
	my %misc = % { $cache{misc} };

	my $feat_key = join("_", map {$_ . ":" . $featref->{$_}} sort keys %$featref);
	if(my $cached_result = $svm_cache->{$feat_key}) {
		return $cached_result;
	}
	
	# make sure feature vector lines up with support vectors
	my %features = % { $featref };	
	my @svm = @{ $cache{svm} };
	my $nrow = scalar @svm;
	my @header = @ { $svm[0] };
	my $ncol = scalar @header;
	my @featnames = keys %features;
	
	# map features to corresponding index in header of support vector matrix
	my @idx = map { get_index($_, \@featnames) } @header[0..$ncol-2];
	
	# center and scale data for SVM
	my %center = % { $cache{center} };
	my %scale = % { $cache{scale} };
	my @sorted = map { $features{$featnames[$_]} } @idx;
	my @cen = map { $center{$_} } @header[0..$ncol-2];
	my @sca = map { $scale{$_} } @header[0..$ncol-2];
	my @centered = pairwise { $a - $b } @sorted, @cen;
	my @scaled = pairwise { $a / $b } @centered, @sca;
	
	# compute SVM decision function
	my $margin = 0;
	my $gamma = ${misc}{gamma};
	for (my $i=1; $i < $nrow; $i++) {
	   my $row = $svm[$i];
	   my $alpha = $row->[-1];
	   my $dot;
	   if ($kernel eq "radial") {
	   	# we can just pass all of the list referenced by $row as the for loop in rbf()
	   	# will only use up to the length of @scaled
	   	$dot = rbf(\@scaled, $row, $gamma);
	   } else {
	   	my @sv = @$row[0..$ncol-2];
	   	$dot = sum(pairwise { $a * $b } @scaled, @sv);
	   }
	   $margin = $margin + ($alpha * $dot);
	};
	$margin = $margin - ${misc}{rho};

	$svm_cache->{$feat_key} = $margin;

	return $margin;
}


# radial basis function kernel
sub rbf {
	my ($v1, $v2, $gamma) = @_[0..2];

	my $k = 0;
	for(my $i=0; $i<@$v1; $i++) {
		$k += ($v1->[$i] - $v2->[$i]) ** 2;
	}

	return exp(-$gamma * $k);
}


# transforms SVM decision function output into a probability
sub svm_logreg {
	my ($x, $cacheref) = @_[0..1];
	my %cache = % { $cacheref };
	my %misc = % { $cache{misc} };
	my $logit = $misc{probA} * $x + $misc{probB};
	return 1 / (1 + exp(-$logit));
}

# helper function, finds index for first occurence of a given element in a given list
sub get_index {
	my ($elem, $lref) = @_[0..1];
	my @list = @$lref;
	my $idx = firstidx { $_ eq $elem } @list;
	return $idx;
}

1;