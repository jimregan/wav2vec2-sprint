#!/usr/bin/perl
# Scrapes sounds from fuaimeanna.ie
# Creates a set of labels in 'label/' (which must exist)
# along with three files:
# run-wget.sh, which downloads the sounds to mp3/ (it creates it)
# run-ffmpeg.sh, which converts the sounds in mp3/ to wav files in wav/
# all-fuaimeanna-data.tsv, which contains all of the data

use warnings;
use strict;
use utf8;

use URI;
use Web::Scraper;
use Data::Dumper;

binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");
open(WGET, ">", "run-wget.sh");
binmode(WGET, ":utf8");
open(FFMPEG, ">", "run-ffmpeg.sh");
binmode(FFMPEG, ":utf8");
open(ALL, ">", "all-fuaimeanna-data.tsv");
binmode(ALL, ":utf8");

my $phones = scraper {
    process 'div[class="friotal"]', 'sounds[]' => scraper {
        process 'span[class="ortho"]', 'orth' => 'TEXT';
        process 'span[class="taifead"] span[class="player"] a audio source', 'sounds[]' => '@src';
        process 'span[class="phonological"]', 'dialects[]' => scraper {
            process 'a[class="phoneme"]', 'phonemes[]' => 'TEXT';
        };
    };
};

if(! -d "label") {
    die "Directory 'label' does not exist\n";
}

# write shell headers to output files
print WGET "#!/bin/sh\n";
print WGET "mkdir mp3\n";

print FFMPEG "#!/bin/sh\n";
print FFMPEG "mkdir wav\n";

# write .tsv header
print ALL "Orthographic\t";
print ALL "Audio (Gaoth Dobhair)\tIPA (Gaoth Dobhair)\t";
print ALL "Audio (Ceathrú Rua)\tIPA (Ceathrú Rua)\t";
print ALL "Audio (Corca Dhuibhne)\tIPA (Corca Dhuibhne)\n";

for my $i (1..77) {
    my $res = $phones->scrape(URI->new("http://www.fuaimeanna.ie/en/Recordings.aspx?Page=$i"));

    for my $sound (@{$res->{'sounds'}}) {
        my $word = $sound->{'orth'};
        $word =~ s/^<//;
        $word =~ s/>$//;
        if($#{$sound->{'sounds'}} != 2 && $#{$sound->{'dialects'}} != 2) {
            print STDERR "Error reading <$word> on page $i";
        }
        print ALL "$word\t";
        for my $j (0..2) {
            my $sound_raw = ${$sound->{'sounds'}}[$j];
            my $phones_raw = join(' ', @{${$sound->{'dialects'}}[$j]->{'phonemes'}});
            # put everything in a tsv file first, because it doesn't make sense to hammer their server again
            print ALL $sound_raw . "\t" . $phones_raw;
            print ALL "\t" unless ($j == 2);

            my $sound_base = $sound_raw;
            $sound_base =~ s!/sounds/!!;
            $sound_base =~ s/\.mp3//;
            
            my $phones_out = $phones_raw;
            # discard word boundary
            $phones_out =~ s/ \# / /g;
            $phones_out =~ s/\.//g;
            $phones_out =~ s/ˈ//g;
            $phones_out =~ s/\s+/ /g;
            $phones_out =~ s/^ //;
            $phones_out =~ s/ $//;
            
            # write the script line
            print WGET "wget http://www.fuaimeanna.ie$sound_raw -O mp3/$sound_base.mp3\n";
            print FFMPEG "ffmpeg -i \"mp3/$sound_base.mp3\" -acodec pcm_s16le -ac 1 -ar 16000 wav/$sound_base.wav\n";
            
            # write the phones to the label file
            my $label_file = "label/$sound_base.phones";
            open(OUT, ">", $label_file);
            binmode(OUT, ":utf8");
            print OUT "$phones_out";
            close(OUT);
        }
        # add a newline to the tsv file
        print ALL "\n";
    }
}