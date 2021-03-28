#!/usr/bin/perl
# License: Apache-2.0

my %dialects = (
	'U' => 'ulster',
	'C' => 'connacht',
	'M' => 'munster'
);

my $base='.';
if($ARGV[0] && $ARGV[0] ne '') {
	$base = $ARGV[0];
	$base =~ s/\/$//;
}

#print '"path","dialect","sentence"' . "\n";
while(<STDIN>) {
	chomp;
	my $file = $base . '/' . $_;
	my $dialect = '';
	if($file =~ /\/Can([UMC])\//) {
		my $abbr = $1;
		$dialect = $dialects{$abbr};
	} else {
		next;
	}
	$file =~ s/'/\\'/g;
	my @parts = split/\//, $file;
	my $transcript = $parts[$#parts];
	$transcript =~ s/\.mp3$//;
	print "{\"path\": \"$file\", \"accent\": \"$dialect\", \"sentence\": \"$transcript\"}\n";
}
