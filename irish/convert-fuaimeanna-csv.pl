#!/usr/bin/perl

use warnings;
use strict;
use utf8;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $base='.';
if($ARGV[0] && $ARGV[0] ne '') {
	$base = $ARGV[0];
	$base =~ s/\/$//;
}

my %cr_files = (
	'mo shmidiú' => 'mo chuid smidiú',
	'mo shmior' => 'mo chuid smior',
	'mo shmólach' => 'mo smólach',
	'shmachtaigh' => 'smachtaigh',
	'shmaoinigh' => 'smaoinigh',
	'shmear' => 'smear',
	'deamhain' => 'diabhail'
);

print '"path", "dialect", "transcript"' . "\n";
while(<STDIN>) {
	chomp;
	my @line = split/\t/;
	next if($line[0] eq 'Orthographic');
	my $text = $line[0];
	print "\"$base$line[1]\", \"ulster\", \"$text\"\n";
	print "\"$base$line[3]\", \"connacht\","; 
	if(exists $cr_files{$text}) {
		print "\"$cr_files{$text}\"\n";
	} else {
		print "\"$text\"\n";
	}
	print "\"$base$line[5]\", \"munster\", \"$text\"\n";
}