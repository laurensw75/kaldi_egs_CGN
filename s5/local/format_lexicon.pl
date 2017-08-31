#!/usr/bin/perl

#
# This script takes the xml file containing CGN ortho/phonetic transcriptions of words (cgnlex.lex)
# and creates a standard lexicon.txt file with words followed by phonetic transcriptions.
#

if ($ARGV[0]=='vl') {
	$lang="pronflnorm";
} else {
	$lang="pron".$ARGV[0]."norm";
}

while (<STDIN>) {
    chop;
    if (m/^\s+<orth>(\S+)<\/orth>/) {
        $orth=$1;
    } elsif (m/^\s+<$lang>(\S+)<\/$lang>/) {
        $trans=join(" ", split(//, $1));
        $trans=~s/ (\+|:|~)/$1/g;
        if ($orth ne '') {
            $wordlist{lc($orth)}=$trans;
        }
    }
}
foreach $word (sort keys %wordlist) {
    print "$word\t$wordlist{$word}\n";
}
