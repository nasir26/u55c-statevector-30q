#!/bin/bash
# Runs the flagship campaign parts on two cards in parallel (author-approved cards only).
cd "${WS:-$PWD}"
export FLAG=sv_e16w_m4
( bench/fpga/campaign.sh 0000:41:00.1 main; bench/fpga/campaign.sh 0000:41:00.1 energy; echo CARD41_DONE ) > logs/campaign_card41.log 2>&1 &
( sed 's/0000:41:00.1/0000:a1:00.1/' /dev/null; bench/fpga/campaign.sh 0000:a1:00.1 ablate; echo CARDA1_DONE ) > logs/campaign_carda1.log 2>&1 &
wait
