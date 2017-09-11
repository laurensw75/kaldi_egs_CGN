#!/usr/bin/perl
while(<STDIN>) {
    chop;
    s/'t/het/g;
    s/\<\S+\>//g;
    s/e\.//g;
    s/_/ /g;
    s/\s+/ /g;
    print "$_\n";
}