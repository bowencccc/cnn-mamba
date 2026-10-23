#!/usr/bin/env perl
use strict;
use warnings;

# Convert one-row-per-base joint-frame probabilities into the five emission
# columns consumed by UniAnn.  This preserves UniAnn's coordinate/frame
# mapping while making the probability transformation an explicit ablation.

my ($fasta, $method, $scale_parameter, $coding_bias) = @ARGV;
die "usage: $0 FASTA METHOD [SCALE_PARAMETER] [CODING_BIAS]\n"
    unless defined $fasta && defined $method;

my %valid_method = map { $_ => 1 } qw(
    raw_double_gate
    binary_t6
    binary_t7
    softmax_t6
    softmax_t7
    raw_joint
    joint_softmax
    emission_affine
);
die "unknown method: $method\n" unless $valid_method{$method};
if ($method eq 'joint_softmax' || $method eq 'emission_affine') {
    die "scale parameter is required for $method\n"
        unless defined $scale_parameter && $scale_parameter > 0;
    $coding_bias = 0 unless defined $coding_bias;
}

open my $fasta_fh, '<', $fasta or die "cannot open $fasta: $!\n";
my $sequence = '';
while (my $line = <$fasta_fh>) {
    chomp $line;
    next if substr($line, 0, 1) eq '>';
    $sequence .= uc $line;
}
close $fasta_fh;
my $sequence_length = length $sequence;
print STDERR "Loaded sequence of $sequence_length bases\n";

my %stops;
while ($sequence =~ /TAA|TAG|TGA/g) {
    $stops{$-[0]} = 1;
}
print STDERR "Found stop codons\n";

my $temperature = $method =~ /_t(\d+)$/ ? 0 + $1 : 1;
my $n = 0;
my @coding_frame;
my $intron_stop = -2;

while (my $line = <STDIN>) {
    chomp $line;
    next if $line eq '';
    my @fields = split /\t/, $line;
    if ($fields[0] eq 'chrom') {
        next;
    }
    die "input row has fewer than seven columns at line " . ($n + 2) . "\n"
        if @fields < 7;
    die "non-contiguous position at row " . ($n + 1) . ": $fields[1]\n"
        unless $fields[1] =~ /^\d+$/ && $fields[1] == $n + 1;

    my @probability = map { 0 + $_ } @fields[3 .. 6];
    for my $value (@probability) {
        die "probability outside [0,1] at position " . ($n + 1) . "\n"
            unless $value >= 0 && $value <= 1;
    }

    if ($method =~ /^binary_/) {
        @probability = map {
            1 / (1 + (((1 - $_) / ($_ + 1e-10)) ** (1 / $temperature)))
        } @probability;
    }
    elsif ($method =~ /^softmax_/) {
        my @powered = map { ($_ + 1e-30) ** (1 / $temperature) } @probability;
        my $total = 0;
        $total += $_ for @powered;
        @probability = map { $_ / $total } @powered;
    }
    elsif ($method eq 'joint_softmax') {
        my @powered = map {
            ($_ + 1e-30) ** (1 / $scale_parameter)
        } @probability;
        for my $index (0 .. 2) {
            $powered[$index] *= exp($coding_bias);
        }
        my $total = 0;
        $total += $_ for @powered;
        @probability = map { $_ / $total } @powered;
    }

    my ($frame0, $frame1, $frame2, $noncoding) = @probability;
    my @frame_probability = ($frame0, $frame1, $frame2);
    for my $i (0 .. 2) {
        my $emission;
        if ($method eq 'emission_affine') {
            $emission = $scale_parameter
                * log(exp(1) * $frame_probability[$i] + 1e-10)
                + $coding_bias;
        }
        else {
            my $emission_probability = $frame_probability[$i];
            $emission_probability *= 1 - $noncoding
                unless $method eq 'raw_joint'
                    || $method eq 'joint_softmax';
            $emission = log(exp(1) * $emission_probability + 1e-10);
        }
        $coding_frame[($n - $i - 1) % 3] = $emission;
    }
    my $noncoding_score = log(exp(1) * $noncoding + 1e-10);
    $noncoding_score *= $scale_parameter if $method eq 'emission_affine';
    my $intron_score;
    if ($stops{$n - 2}) {
        $coding_frame[($n - 2) % 3] = -1000000;
        $intron_score = $intron_stop;
    }
    else {
        $intron_score = $noncoding_score;
    }
    printf "%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n",
        $n, $noncoding_score, @coding_frame, $intron_score,
        substr($sequence, $n, 1);
    $n++;
}

die "probability rows ($n) do not match FASTA length ($sequence_length)\n"
    unless $n == $sequence_length;
print STDERR "Done processing $n scores with method=$method";
print STDERR " scale=$scale_parameter bias=$coding_bias"
    if defined $scale_parameter;
print STDERR "\n";
