#!/usr/bin/perl
# License: Apache-2.0

use warnings;
use strict;
use utf8;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

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

my %fixU = (
    'éigin' => 'inteacht',
    'a Dhia' => 'Dhia',
    'cóngar' => 'cógar',
    'cóngarach' => 'cógarach',
    'cóngaracht' => 'cógaracht',
    'teanga' => 'teangaí',
    'gnóthach' => 'gnítheach',
    'athghafa' => 'athghoite',
    'geata' => 'geafta',
);
my %fixC = (
    'éigin' => 'eicint',
    'zúmáil' => 'súmáil',
    'zúm' => 'súm',
    'zipeáil' => 'sipeáil',
    'zip' => 'sip',
    'agam' => "a'am",
    'agat' => "a'at",
#    'conga' => 'cunga',
#    'cóngar' => 'cúngar',
#    'cóngarach' => 'cúngarach',
#    'cóngaracht' => 'cúngaracht',
    'pionós' => 'píonós',
    'feidhmiú teanga' => 'feidhmiú teangan',
    'gnóthach' => 'gnathach',
    'bláthanna' => 'bláthannaí',
    'ceap bláthanna' => 'ceap bláthannaí',
    'ceapach bláthanna' => 'ceapach bláthannaí',
);
my %fixM = (
    'éigin' => 'éigint',
    'fáil' => 'fáilt',
    'a fhad le' => 'fad le',
    'dearmad' => 'dearúd',
    'zónáilte' => 'zónáilthe',
    'seinn' => 'seinm',
    'pionós' => 'píonós',
    'feidhmiú teanga' => 'feidhmiú teangan',
    'teorainn' => 'teora',
);

my %fixAll = (
    'praiticiúil' => 'praicticiúil',
);

for my $k (keys %fixAll) {
    $fixM{$k} = $fixAll{$k};
    $fixC{$k} = $fixAll{$k};
    $fixU{$k} = $fixAll{$k};
}

sub get_replacement {
    my $word = $_[0];
    my $dialect = $_[1];
    if ($dialect eq 'M') {
        if(exists $fixM{$word}) {
            return $fixM{$word};
        } else {
            return $word;
        }
    }
    if ($dialect eq 'U') {
        if(exists $fixU{$word}) {
            return $fixU{$word};
        } else {
            return $word;
        }
    }
    if ($dialect eq 'C') {
        if(exists $fixC{$word}) {
            return $fixC{$word};
        } else {
            return $word;
        }
    }
}

sub irish_lc {
    my $text = shift;
    $text =~ s/^([nt])([AEIOUÁÉÍÓÚ].*)/$1-$2/;
    return lc($text);
}

sub tolower {
    my $text = $_[0];
    my @pieces = split / /, $text;
    my @lcpieces = map { irish_lc($_) } @pieces;
    return join(" ", @lcpieces);
}

sub clean {
    my $in = shift;
    my $lwr = tolower($in);
    $lwr =~ s/[!]//g;
    return $lwr;
}

my %empty_list = (
    'www.teanglann.ie/CanC/boladh coirp.mp3' => 1,
    'www.teanglann.ie/CanC/buille adhmaid.mp3' => 1,
    'www.teanglann.ie/CanM/fiabhras faireoige.mp3' => 1,
);

#print '"path","dialect","sentence"' . "\n";
while(<STDIN>) {
    chomp;
    my $file = $_;
    my $dialect = '';
    my $dialect_name = '';
    if($file =~ /\/Can([UMC])\//) {
        $dialect = $1;
        $dialect_name = $dialects{$dialect};
    } else {
        next;
    }
    $file =~ s/'/\\'/g;
    my @parts = split/\//, $file;
    my $url = '';
    if($file =~ /.*(www.*)$/) {
        $url = $1;
        next if(exists $empty_list{$url});
    }
    my $text = $parts[$#parts];
    $text =~ s/\.mp3$//;
    my $outtext = $text;
    if(/éigin$/) {
        if($dialect ne 'M' || ($dialect eq 'M' && ($text ne 'éigin' && $text ne 'am éigin'))) {
            my $replacement = get_replacement('éigin', $dialect);
            $text =~ s/éigin/$replacement/;
            $outtext = clean("$text");
        } else {
            $outtext = clean("$text");
        }
    } else {
        $outtext = clean(get_replacement($text, $dialect));
    }
    print "{\"path\": \"$file\", \"accent\": \"$dialect_name\", \"sentence\": \"$outtext\"}\n";
}
