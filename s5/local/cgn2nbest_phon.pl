#!/usr/bin/perl

while(<STDIN>) {
    chop;
    @parts=split(/\t/);
    # do this twice to catch overlapping matches
    for my $t (0..1) {
        $parts[1]=" $parts[1] ";
        $parts[1]=~s/ J / n j /g;
        $parts[1]=~s/ Y / U /g;
        $parts[1]=~s/ 2 / EU /g;
        $parts[1]=~s/ E\+ / EI /g;
        $parts[1]=~s/ Y\+ / UI /g;
        $parts[1]=~s/ A\+ / AU /g;
        $parts[1]=~s/ E: / E2 /g;
        $parts[1]=~s/ Y: / U /g;
        $parts[1]=~s/ O: / O /g;
        $parts[1]=~s/ A~ / A n /g;
        $parts[1]=~s/ E~ / E n /g;
        $parts[1]=~s/ O~ / O /g;
        $parts[1]=~s/ Y~ / U m /g;
        $parts[1]=~s/^\s+|\s+$//g;
    }
    print "$parts[0]\t$parts[1]\n";
}