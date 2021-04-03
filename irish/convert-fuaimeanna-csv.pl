#!/usr/bin/perl
# License: Apache 2.0

use warnings;
use strict;
use utf8;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $FULL=0;

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
my %empty = (
	'/sounds/gob_i3_s3.mp3' => 1,
	'/sounds/iioctha_i3_s3.mp3' => 1,
	'/sounds/mo_shuiiochaan_i3_s3.mp3' => 1,
	'/sounds/riail_i3_s3.mp3' => 1
);
if($FULL) {
	print '"path", "accent", "sentence"' . "\n";
} else {
	print '"path","sentence"' . "\n";
}
while(<STDIN>) {
	chomp;
	my @line = split/\t/;
	next if($line[0] eq 'Orthographic');
	my $text = $line[0];
	next if($line[0] eq "d'fhág");
	if($FULL) {
		if($text eq 'Gaeilge') {
			print "\"$base$line[1]\",\"ulster\",\"gaeilic\"\n";
			print "\"$base$line[3]\",\"connacht\",\"gaeilge\"\n";
			print "\"$base$line[5]\",\"munster\",\"gaelainn\"\n";
			next;
		}
		print "\"$base$line[1]\",\"ulster\",\"$text\"\n";
		print "\"$base$line[3]\",\"connacht\","; 
		if(exists $cr_files{$text}) {
			print "\"$cr_files{$text}\"\n";
		} else {
			print "\"$text\"\n";
		}
		print "\"$base$line[5]\",\"munster\",\"$text\"\n";
	} else {
		if($text eq 'Gaeilge') {
			print "\"$base$line[1]\",\"gaeilic\"\n";
			print "\"$base$line[3]\",\"gaeilge\"\n";
			print "\"$base$line[5]\",\"gaelainn\"\n";
			next;
		}
		print "\"$base$line[1]\",\"$text\"\n";
		if(!exists $empty{$line[3]}) {
			print "\"$base$line[3]\",";
			if(exists $cr_files{$text}) {
				print "\"$cr_files{$text}\"\n";
			} else {
				print "\"$text\"\n";
			}
		}
		print "\"$base$line[5]\",\"$text\"\n";
	}
}
