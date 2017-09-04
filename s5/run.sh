#!/bin/bash

#
# This script was adapted from the WSJ/s5 example by LvdW
# You need Corpus Gesproken Nederlands to use this.
# This is available from TST-centrale: http://tst-centrale.org/nl/tst-materialen/corpora/corpus-gesproken-nederlands-detail
#
# The script can train both studio and telephone models (automatically detected from comp)
# We keep these separate as that works best for GMM/HMM. Before follow-up training
# with nnet models, the two training sets get combined again.
#
# By default a fully functioning set of models is created using only CGN. Better performance may be had by
# using more material for the language model and by extending your lexicon.
#

stage=0
train=true	# set to false to disable the training-related scripts
				# note: you probably only want to set --train false if you
				# are using at least --stage 1.
decode=true	# set to false to disable the decoding-related scripts.

. ./cmd.sh	## You'll want to change cmd.sh to something that will work on your system.
           	## This relates to the queue.
 
[ ! -e steps ] && ln -s ../../wsj/s5/steps steps
[ ! -e utils ] && ln -s ../../wsj/s5/utils utils
          	
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

cgn=/home/laurensw/Data/CGN			# point this to CGN
lang="nl"
comp="a;b;c;d;f;g;h;i;j;k;l;m;n;o"
nj=30;
decode_nj=10;

if [ $stage -le 0 ]; then
  # data preparation.

  # the script detects if a telephone comp is used and splits this into a separate set
  # later, studio and telephone speech can be combined for NNet training
  local/cgn_data_prep.sh $cgn $lang $comp || exit 1;

  # the text in cleaned.gz is used to train the lm..
  cat data/train_s/text data/train_t/text | cut -d' ' -f2- | gzip -c >data/local/dict_nosp/cleaned.gz
  # you are encouraged to use your own additional data for training and tune the pruning
  # in the following script accordingly
  local/cgn_train_lms.sh --dict-suffix "_nosp"
  local/cgn_format_local_lms.sh --lang-suffix "_nosp"

  for x in train_s dev_s; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$x || exit 1;
    steps/compute_cmvn_stats.sh data/$x || exit 1;
  done

  for x in train_t dev_t; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_t.conf data/$x || exit 1;
    steps/compute_cmvn_stats.sh data/$x || exit 1;
  done
	
  # Make subsets with 5k random utterances from train.
  # using only the shortest ones doesn't work as these are too similar
  for x in train_s train_t; do
    utils/subset_data_dir.sh data/$x 5000 data/${x}_5k || exit 1;
  done
	
  # Make sure dev has text_ref
  cp data/dev_s/text data/dev_s/text_ref
  cp data/dev_t/text data/dev_t/text_ref
	
  # do a final cleanup
  for x in train_s train_t dev_s dev_t; do
    utils/fix_data_dir.sh data/$x
  done
fi

if [ $stage -le 1 ]; then
  # monophone
  if $train; then
    for x in train_s train_t; do
      steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
        data/${x}_5k data/lang_nosp exp/$x/mono0a || exit 1;
    done
  fi
	
  if $decode; then
    for x in s t; do
      utils/mkgraph.sh data/lang_nosp_test_tgpr exp/train_${x}/mono0a exp/train_${x}/mono0a/graph_nosp_tgpr
      nspk=$(wc -l <data/dev_${x}/spk2utt)
      [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
        exp/train_${x}/mono0a/graph_nosp_tgpr data/dev_${x} exp/train_${x}/mono0a/decode_nosp_tgpr
    done
  fi
fi

if [ $stage -le 2 ]; then
  # tri1
  if $train; then
    for x in train_s train_t; do
      steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/${x}_5k data/lang_nosp exp/$x/mono0a exp/$x/mono0a_ali || exit 1;
      steps/train_deltas.sh --cmd "$train_cmd" 2000 10000 \
        data/${x}_5k data/lang_nosp exp/$x/mono0a_ali exp/$x/tri1 || exit 1;
    done
  fi
	
  if $decode; then
    for x in s t; do
      utils/mkgraph.sh data/lang_nosp_test_tgpr exp/train_${x}/tri1 exp/train_${x}/tri1/graph_nosp_tgpr || exit 1;
      nspk=$(wc -l <data/dev_${x}/spk2utt)
      [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
        exp/train_${x}/tri1/graph_nosp_tgpr data/dev_${x} exp/train_${x}/tri1/decode_nosp_tgpr || exit 1;
      # due to the following command not accepting the scoring options, we made --combine false the default for local/score.sh
      steps/lmrescore.sh --mode 4 --cmd "$decode_cmd" \
        data/lang_nosp_test_{tgpr,tg} data/dev_${x} \
        exp/train_${x}/tri1/decode_nosp_tgpr \
        exp/train_${x}/tri1/decode_nosp_tgpr_tg || exit 1;
    done
  fi
fi

if [ $stage -le 3 ]; then
  # tri2
  if $train; then
    for x in train_s train_t; do
      steps/align_si.sh --nj 10 --cmd "$train_cmd" \
        data/$x data/lang_nosp exp/$x/tri1 exp/$x/tri1_ali || exit 1;

      steps/train_lda_mllt.sh --cmd "$train_cmd" \
        --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
        data/$x data/lang_nosp exp/$x/tri1_ali exp/$x/tri2 || exit 1;
    done
  fi

  if $decode; then
    for x in s t; do
      utils/mkgraph.sh data/lang_nosp_test_tgpr exp/train_${x}/tri2 exp/train_${x}/tri2/graph_nosp_tgpr || exit 1;
      nspk=$(wc -l <data/dev_${x}/spk2utt)
      [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" exp/train_${x}/tri2/graph_nosp_tgpr \
        data/dev_${x} exp/train_${x}/tri2/decode_nosp_tgpr || exit 1;
      # compare lattice rescoring with biglm decoding, going from tgpr to tg.
      steps/decode_biglm.sh --nj $nspk --cmd "$decode_cmd" \
        exp/train_${x}/tri2/graph_nosp_tgpr data/lang_nosp_test_{tgpr,tg}/G.fst \
        data/dev_${x} exp/train_${x}/tri2/decode_nosp_tgpr_tg_biglm
      # baseline via LM rescoring of lattices.
      steps/lmrescore.sh --cmd "$decode_cmd" \
        data/lang_nosp_test_tgpr/ data/lang_nosp_test_tg/ \
        data/dev_${x} exp/train_${x}/tri2/decode_nosp_tgpr \
        exp/train_${x}/tri2/decode_nosp_tgpr_tg || exit 1;
      # Demonstrating Minimum Bayes Risk decoding (like Confusion Network decoding):
      mkdir exp/train_${x}/tri2/decode_nosp_tgpr_tg_mbr
      cp exp/train_${x}/tri2/decode_nosp_tgpr_tg/lat.*.gz exp/train_${x}/tri2/decode_nosp_tgpr_tg_mbr;
      local/score_mbr.sh --cmd "$decode_cmd"  \
        data/dev_${x} data/lang_nosp_test_tgpr/ \
        exp/train_${x}/tri2/decode_nosp_tgpr_tg_mbr
    done
  fi
fi

if [ $stage -le 4 ]; then
  # Estimate pronunciation and silence probabilities.
  model=tri2

  # Silprob for normal lexicon.
  for x in s t; do
    steps/get_prons.sh --cmd "$train_cmd" data/train_${x} data/lang_nosp exp/train_${x}/$model || exit 1;
    utils/dict_dir_add_pronprobs.sh --max-normalize true \
      data/local/dict_nosp \
      exp/train_${x}/$model/pron_counts_nowb.txt exp/train_${x}/$model/sil_counts_nowb.txt \
      exp/train_${x}/$model/pron_bigram_counts_nowb.txt data/local/dict_${x} || exit 1

    utils/prepare_lang.sh data/local/dict_${x} \
      "<unk>" data/local/lang_tmp_${x} data/lang_${x} || exit 1;

    for lm_suffix in tg tgpr fgconst; do
      mkdir -p data/lang_${x}_test_${lm_suffix}
      cp -r data/lang_${x}/* data/lang_${x}_test_${lm_suffix}/ || exit 1;
      rm -rf data/lang_${x}_test_${lm_suffix}/tmp
      cp data/lang_nosp_test_${lm_suffix}/G.* data/lang_${x}_test_${lm_suffix}/
    done
  done
fi

if [ $stage -le 5 ]; then
  # From tri2 system, train tri3 which is LDA + MLLT + SAT.
  # now using data/lang as the lang directory (we have now added
  # pronunciation and silence probabilities)

  if $train; then
    for x in s t; do
      steps/align_si.sh --nj 10 --cmd "$train_cmd" \
        data/train_${x} data/lang_${x} exp/train_${x}/tri2 exp/train_${x}/tri2_ali  || exit 1;
      steps/train_sat.sh --cmd "$train_cmd" 5000 80000 \
        data/train_${x} data/lang_${x} exp/train_${x}/tri2_ali exp/train_${x}/tri3 || exit 1;
    done
  fi

  if $decode; then
    for x in s t; do
      utils/mkgraph.sh data/lang_${x}_test_tgpr exp/train_${x}/tri3 exp/train_${x}/tri3/graph_tgpr || exit 1;
      nspk=$(wc -l <data/dev_${x}/spk2utt)
      [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
      steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" \
        exp/train_${x}/tri3/graph_tgpr data/dev_${x} \
        exp/train_${x}/tri3/decode_tgpr || exit 1;
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_${x}_test_{tgpr,fgconst} \
        data/dev_${x} exp/train_${x}/tri3/decode_tgpr{,_fg} || exit 1;
    done
  fi
fi

# It is time to clean up our data a bit
# this takes quite a while.. and is actually only really helpful for the NNet models,
# so if you're not going to make those, you may as well stop here.

if [ $stage -le 7 ]; then  
  for x in s t; do
    steps/cleanup/clean_and_segment_data.sh --nj $nj --cmd "$train_cmd" --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
      data/train_${x} data/lang_${x} exp/train_${x}/tri3 exp/train_${x}/tri3_cleaned_work data/train_${x}_cleaned
  done
fi

if [ $stage -le 8 ]; then
  # Now we're going to recombine our telephone and studio speech to one data dir, and make sure we have alignments for them
  # The telephone speech will be 8khz, so we need to upsample & get new features
  for x in train_t_cleaned dev_t; do
    utils/copy_data_dir.sh data/$x data/${x}_16khz
    rm data/${x}_16khz/feats.scp
    cat data/$x/wav.scp | sed 's/wav -b 16/wav -r 16k -b 16/' >data/${x}_16khz/wav.scp
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc.conf data/${x}_16khz || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_16khz || exit 1;
  done
  cp data/dev_t/text_ref data/dev_t_16khz/
  if [ -d data/train_s_cleaned ] && [ -d data/train_t_cleaned_16khz ]; then
    utils/combine_data.sh data/train_cleaned data/train_s_cleaned data/train_t_cleaned_16khz
  elif [ -d data/train_s_cleaned ]; then
    utils/copy_data_dir.sh data/train_s_cleaned data/train_cleaned
  elif [ -d data/train_t_cleaned_16khz ]; then
    utils/copy_data_dir.sh data/train_t_cleaned_16khz data/train_cleaned
  fi
fi

if [ $stage -le 9 ]; then
  # Do one more pass of sat training.
  if $train; then
    # use studio models for this alignment pass
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train_cleaned data/lang_s exp/train_s/tri3 exp/train_cleaned/tri3_ali_cleaned
    steps/train_sat.sh --cmd "$train_cmd" 5000 80000 \
      data/train_cleaned data/lang_s exp/train_cleaned/tri3_ali_cleaned exp/train_cleaned/tri4 || exit 1;
  fi

  if $decode; then
    utils/mkgraph.sh data/lang_s_test_tgpr exp/train_cleaned/tri4 exp/train_cleaned/tri4/graph_tgpr || exit 1;
    for x in dev_s dev_t_16khz; do
      nspk=$(wc -l <data/$x/spk2utt)
      [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
      steps/decode_fmllr.sh --nj $nspk --cmd "$decode_cmd" \
        exp/train_cleaned/tri4/graph_tgpr data/$x \
        exp/train_cleaned/tri4/decode_${x}_tgpr || exit 1;
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_s_test_{tgpr,fgconst} \
        data/$x exp/train_cleaned/tri4/decode_${x}_tgpr{,_fg} || exit 1;
    done
  fi
fi

# To train nnet models, please run local/chain/run_tdnn.sh

exit 0;
