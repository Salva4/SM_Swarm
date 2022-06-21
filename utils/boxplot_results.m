% Plot accuracies & running times

close all
clear all
clc

%% Table & labels
% Labels
data_reduction = {'Original'; 'Balanced'; 'LDA'; 
    'PCA'; 'Autoencoder'; 'PCA 1%-corrupted'; 
    'PCA 2%-corrupted'; 'PCA 3%-corrupted'; 'PCA small';
    'PCA 1%-corrupted small'; 'PCA 2%-corrupted small'; 'PCA 3%-corrupted small';
    'LDA balanced'; 'PCA balanced'; 'Autoencoder balanced';
};

% Table
% 1. eSPA
eSPA_original = [       % Most repeated W: 100, CL: 10
   0.879498296571376  12.073467756975001
   0.854921864354392  12.895058964975002
   0.861090233549117  10.068309606824998
   0.874112290086956   9.685330663725003
   0.866560913913558   9.848849511225000
   0.855413427596712  10.775723513949998
   0.881693739759921  10.102933937850000
   0.867282048117811   9.283453368950001
   0.876267773679758   9.722826661300001
   0.863412771296860   9.066101175375000
];
eSPA_balanced = [       % Most repeated W: 100, CL: 10
   0.857459561426801   5.709382283049997
   0.875382813907404   4.653698497550001
   0.864758903823848   5.547404134275000
   0.859338585427119   5.906936484225001
   0.856240060844915   4.423950522700001
   0.857636002275924   4.671247201250000
   0.862751593403489   5.291516512474999
   0.858027617900749   4.979366022124998
   0.854070463890970   4.907741029300000
   0.855374238468233   4.608664252749999
];
eSPA_LDA = [
   0.951664903598802   0.036496952675000
   0.953245840631200   0.036129287550000
   0.954452591686083   0.027532133400000
   0.953106718131349   0.034581423125000
   0.948151720204723   0.043413126525000
   0.951420530247951   0.033339462150000
   0.952641695008182   0.025620477200000
   0.950727320406621   0.031238107925000
   0.954762853317343   0.037797484150000
   0.950375367866571   0.031462864625000
];
eSPA_PCA = [
   0.947154651695843   0.063690803150000
   0.948789329548967   0.044567998250000
   0.949059815304379   0.052083361025000
   0.950289057255982   0.057134181225000
   0.951791985761789   0.044276141800000
   0.949700896631587   0.041376469400000
   0.947846482122999   0.047574695800000
   0.946977134013091   0.050068520875000
   0.955579148834802   0.050228753150000
   0.949370547842102   0.041366641400000
];
eSPA_autoencoder = [
   0.840649961907840   0.287319327075000
   0.839241974561879   0.285962989725000
   0.830022617207541   0.258235058425000
   0.829415802744444   0.252572213850000
   0.828410746506410   0.242078836100000
   0.835869211207648   0.255800568800000
   0.827220703949316   0.305181567200000
   0.846662655992102   0.280214039750000
   0.826983341410723   0.262742945550000
   0.827106904323878   0.266068649325000
];
eSPA_PCA_corrupted1 = [
   0.928668047135452   0.148586526800000
   0.926604126223617   0.135151373150000
   0.931039559076111   0.149379461550000
   0.907998729300549   0.132175486050000
   0.920242896683947   0.147708503625000
   0.921914512906185   0.138602622625000
   0.907614060530154   0.138025120325000
   0.907921508041066   0.136030385375000
   0.909300350247536   0.145677701825000
   0.924153447346617   0.174602416700000
];
eSPA_PCA_corrupted2 = [
   0.921389460553010   0.062235230875000
   0.924448926706282   0.048329117250000
   0.928531301988985   0.051315686125000
   0.929738674360992   0.055133366350000
   0.929536317030555   0.051163428150000
   0.930031822107577   0.042885255275000
   0.924204593815986   0.046804352900000
   0.924790953961321   0.048305057300000
   0.936573507560750   0.058746120525000
   0.928612696780865   0.054003971075000
];  
eSPA_PCA_corrupted3 = [
   0.912645197577197   0.054355734175000
   0.912903939977191   0.049160798075000
   0.918811652281134   0.052603517775000
   0.916385637730551   0.056673603675000
   0.916093658998253   0.047732415525000
   0.915841279449545   0.043766021050000
   0.916056510946310   0.049503216050000
   0.911879522314130   0.059486977575000
   0.919918641675367   0.059723274850000
   0.914202935097280   0.047308509775000
];
eSPA_PCA_small = [
   0.940781697150678   0.068273505875000
   0.953335610251025   0.056437632925000
   0.935443846838181   0.071281769125000
   0.942342961049502   0.050780634225000
   0.919255863441123   0.052802388675000
   0.927677768669346   0.071287149850000
   0.938950944246058   0.058845999450000
   0.920179668285372   0.049932286100000
   0.934675416950820   0.048788280175000
   0.917884726701398   0.067128119400000
];
eSPA_PCA_corrupted1_small = [
   0.926654787022040   0.070263421500000
   0.944351151696217   0.057536478750000
   0.925114750445633   0.080348986500000
   0.929395242985207   0.049207414300000
   0.908135305964647   0.059927577525000
   0.908445347572952   0.052645338425000
   0.926061533961996   0.053518104900000
   0.917052188199741   0.059658156575000
   0.918801567464474   0.063784170375000
   0.899882681662343   0.073362330825000
];
eSPA_PCA_corrupted2_small = [
   0.925097106410176   0.070118196250000
   0.937205700021308   0.052965664150000
   0.917542390246688   0.065764906800000
   0.909961374060506   0.053140620950000
   0.908495441339884   0.057471652575000
   0.909320079958378   0.059715912300000
   0.917313461606636   0.067106934900000
   0.904190258420441   0.054440945975000
   0.912348507523605   0.058093038650000
   0.902991592968665   0.073757052300000
];
eSPA_PCA_corrupted3_small = [
   0.913120923649080   0.030804939575000
   0.909106114023389   0.018934425700000
   0.909498011654925   0.021556969875000
   0.915412816596455   0.021399715775000
   0.916498915729039   0.021638584950000
   0.906741071233105   0.018463294650000
   0.921121768438577   0.024222266975000
   0.916692753702091   0.024546354475000
   0.927828344263739   0.021232641750000
   0.910631758942601   0.022294808175000
];
eSPA_LDA_balanced = [
   0.946500486811825   0.058940340500000
   0.954180651818702   0.060050732250000
   0.947987612552108   0.047937966275000
   0.954403892886364   0.039152124550000
   0.955156507738959   0.054720126550000
   0.958121809359738   0.080934361625000
   0.953652926635685   0.077104241100000
   0.952199665601522   0.051021450475000
   0.953709278767593   0.075466297500000
   0.958169158952449   0.055054289300000
];
eSPA_PCA_balanced = [
   0.947851271084642   0.095725122125000
   0.932541987060969   0.091957419625000
   0.935406388566697   0.094419375075000
   0.928398443256330   0.084473203250000
   0.935860767870724   0.093989103550000
   0.937529634429676   0.088324500900000
   0.939820054938000   0.079325650425000
   0.945591897948412   0.079739360550000
   0.933849628147628   0.088020175050000
   0.920718411266714   0.085457253325000
];
eSPA_autoencoder_balanced = [
   0.851585418089169   0.072562448450000
   0.861215538847118   0.076391117525000
   0.855994930611961   0.083430436150000
   0.846681647815783   0.063675612175000
   0.852533064667477   0.071800807650000
   0.851176091732389   0.069584203875000
   0.850246936552654   0.083935923525000
   0.848033196476490   0.071937182125000
   0.852702488076387   0.051015011100000
   0.840688345596962   0.069413843275000
];


% 2. Random Forest
RF_original = [
	0.8620197362751012 0.6240091323852539
	0.8551055843247636 0.9451498985290527
	0.869385456885457 0.9184937477111816
	0.8779426683748219 0.9336099624633789
	0.8568069592252109 0.9231698513031006
	0.87428783512357 0.9221906661987305
	0.8692102413451636 0.915132999420166
	0.8748723827775211 0.937769889831543
	0.8656418628443313 0.9166181087493896
	0.8586090333684647 0.930077075958252
];
RF_balanced = [
	0.9272678634650886 2.174765110015869
	0.9306992259968963 2.418659210205078
	0.9235372571576663 2.3922040462493896
	0.9270595425613577 2.532304048538208
	0.9177139696368072 2.4775640964508057
	0.9203126234767903 2.3828113079071045
	0.9307678875418426 2.589219808578491
	0.920367806718944 2.45749831199646
	0.922206919330318 2.660720109939575
	0.9183847614526466 2.415781259536743
];
RF_LDA = [
	0.8789679333271769 0.765413761138916
	0.8793128732713611 0.7408058643341064
	0.8750373753745708 0.742595911026001
	0.8771842279231442 0.7307770252227783
	0.8820431046774032 0.7360622882843018
	0.875737323626267 0.732835054397583
	0.8651975422364838 0.725567102432251
	0.8755490767735665 0.7224118709564209
	0.8696906907400258 0.7256419658660889
	0.8681784429431749 0.7212378978729248
];
RF_PCA = [
	0.8750072144242312 1.6320858001708984
	0.8745299315025242 1.672969102859497
	0.8562645212520872 1.696471929550171
	0.8749365576951783 1.7634270191192627
	0.8748178842948661 1.670546054840088
	0.8678371045264186 1.7085700035095215
	0.868088263152491 1.6104040145874023
	0.8709037900874634 1.6510887145996094
	0.8678537691290383 1.6135878562927246
	0.8630975653370431 1.4889459609985352
];
RF_autoencoder = [
	0.8833600747991505 2.7037742137908936
	0.8822662245294746 2.5347068309783936
	0.8748266027704343 2.5046708583831787
	0.8870321796430171 2.4756298065185547
	0.8875883128014435 2.4784200191497803
	0.8823140200874426 2.4683949947357178
	0.880639544888601 2.487082004547119
	0.8776222870100421 2.498077154159546
	0.8863365478605869 2.473146677017212
	0.8778704200422497 2.4503209590911865
];
RF_PCA_corrupted1 = [
	0.8701923076923077 2.986574172973633
	0.8720092592900325 2.9625308513641357
	0.8612303401445018 4.413616895675659
	0.8718145565436206 3.769725799560547
	0.8729868942245762 3.4685990810394287
	0.869099354872669 2.860707998275757
	0.8653971017594497 2.7958080768585205
	0.8747068351149984 2.4742519855499268
	0.8663456246560264 2.8519601821899414
	0.8620037400399463 2.740712881088257
];
RF_PCA_corrupted2 = [
	0.868675835718903 1.8955419063568115
	0.8720092592900325 1.6744179725646973
	0.8606733857922926 1.6507718563079834
	0.871283560446122 1.6521668434143066
	0.8708768922415842 1.6662352085113525
	0.8682824595109402 1.7605159282684326
	0.8687657532721254 1.6885979175567627
	0.8699578879170715 1.6944267749786377
	0.8732756142238725 1.659780740737915
	0.8580103229951842 1.6605761051177979
];  
RF_PCA_corrupted3 = [
	0.8733637685843569 2.9863240718841553
	0.8725214947804326 2.8775222301483154
	0.857623922853959 2.78436017036438
	0.8758258162199049 2.733696937561035
	0.8725661229553466 2.709017276763916
	0.8676208009974636 2.7036311626434326
	0.8630154022353559 2.740665912628174
	0.8727243278263687 2.70505690574646
	0.866872366291821 2.7156660556793213
	0.8600773466932167 2.6934030055999756
];
RF_PCA_small = [
	0.8808568406832684 0.8455250263214111
	0.8814018841308479 0.8031001091003418
	0.8778884929935559 0.776500940322876
	0.8748097739138578 0.7773392200469971
	0.867305094639477 0.767427921295166
	0.8721132148649602 0.7614588737487793
	0.8696066375386166 0.7603662014007568
	0.8732971894844703 0.7632677555084229
	0.8673104187463417 0.7678530216217041
	0.8622949915929325 0.7654032707214355
];
RF_PCA_corrupted1_small = [
	0.8869114275049326 0.6895921230316162
	0.8890825875871645 0.6106290817260742
	0.8851387966889301 0.6107509136199951
	0.8709350148669002 0.5890100002288818
	0.8781631314875196 0.5925581455230713
	0.8796188562160637 0.6010031700134277
	0.8608113637301148 0.6083669662475586
	0.884150054090115 0.6136329174041748
	0.8741383419046144 0.6373817920684814
	0.8686595551145291 0.6072990894317627
];
RF_PCA_corrupted2_small = [
	0.8911240868127767 2.418332099914551
	0.8847349356307213 2.5038859844207764
	0.8790806949092075 2.3005917072296143
	0.8808730387899559 2.4189610481262207
	0.8755436610277008 2.294337034225464
	0.8771816162470911 2.245333671569824
	0.8644932488552988 2.2643237113952637
	0.8826381582629933 2.396080732345581
	0.8699262792952771 2.2881689071655273
	0.8574850464456021 2.609304904937744
];
RF_PCA_corrupted3_small = [
	0.8884933966120977 0.5512759685516357
	0.893076464384409 0.5185160636901855
	0.8851625052497528 0.5020489692687988
	0.8841970480256066 0.5015561580657959
	0.8755430986682202 0.4978470802307129
	0.8709100204498976 0.4918022155761719
	0.868092632996603 0.4938540458679199
	0.8818316615089714 0.49512791633605957
	0.8745342351572974 0.5046720504760742
	0.8626230050442405 0.49088096618652344
];
RF_LDA_balanced = [
	0.9008350087246796 0.29114699363708496
	0.9139253398312933 0.2856261730194092
	0.895351428056724 0.28377199172973633
	0.9025336394355047 0.28730082511901855
	0.9061919578036567 0.28723692893981934
	0.9042219184144394 0.2860221862792969
	0.9110963224409444 0.2858438491821289
	0.8977919586332015 0.28144407272338867
	0.9017637307238874 0.2822411060333252
	0.9083404541498538 0.28548192977905273
];
RF_PCA_balanced = [
	0.9249084856991123 2.6609458923339844
	0.9307249049483736 2.7632389068603516
	0.9275972712533461 2.555516004562378
	0.9248220494701881 2.6614270210266113
	0.9180883239003961 2.5066030025482178
	0.9217722179197421 2.35587477684021
	0.9323882543456369 2.4796040058135986
	0.9237402374924539 2.349623680114746
	0.9216451109235781 2.3731281757354736
	0.9190604478202389 2.315109968185425
];
RF_autoencoder_balanced = [
	0.9127350289558203 2.430478096008301
	0.9224807763418634 2.3162968158721924
	0.9119478270213551 2.24353289604187
	0.9102187871985448 2.24336314201355
	0.9124765658181961 2.2118351459503174
	0.9129680264260087 2.2027339935302734
	0.92097635248183 2.1987571716308594
	0.9132341832364589 2.2023401260375977
	0.908213860153166 2.1847429275512695
	0.9059609146293219 2.2081122398376465
];




% 3. XGBoost
XGB_original = [
	0.8659428596052745 4.68250298500061
	0.8608519945884356 4.503509044647217
	0.8723616473616473 4.459407806396484
	0.8923100132821953 4.176776885986328
	0.8624051219335473 4.296084403991699
	0.8747847349576456 4.408090114593506
	0.8750832015561832 4.372718095779419
	0.8817116129369095 4.352231979370117
	0.8700364602039594 4.4072771072387695
	0.8607155660426036 4.489264965057373
];
XGB_balanced = [
	0.9209721063146448 6.7996909618377686
	0.9300825361006552 6.50507378578186
	0.9225820062008349 6.816323280334473
	0.9214104009156201 7.280401945114136
	0.9142408707934534 6.511062860488892
	0.9165324556273806 6.957120895385742
	0.9288948754689937 6.940675973892212
	0.9215890090995578 7.087385892868042
	0.9197131537442832 6.881242036819458
	0.9155641269087744 6.547658920288086
];
XGB_LDA = [
	0.8791295364299566 0.0614011287689209
	0.8827271533831638 0.059980154037475586
	0.8729414950803388 0.05997419357299805
	0.8749792079841342 0.05971789360046387
	0.8823475185557261 0.0597689151763916
	0.8715187879764845 0.05961298942565918
	0.8661584647220699 0.059580087661743164
	0.8758665370910269 0.059612274169921875
	0.8692600563500603 0.05972599983215332
	0.8693101535719806 0.05995917320251465
];
XGB_PCA = [
	0.8913161418413519 0.06916117668151855
	0.8920186398880642 0.07159304618835449
	0.8758326743285504 0.07392525672912598
	0.8818096517603906 0.06885623931884766
	0.8889641160361791 0.07261180877685547
	0.8857360159357307 0.07396817207336426
	0.8749651132355685 0.07065773010253906
	0.8878717201166181 0.0727071762084961
	0.8762479566942394 0.0670168399810791
	0.868588816856245 0.06764101982116699
];
XGB_autoencoder = [
	0.8826920191153385 0.2206580638885498
	0.8827784600198745 0.22304487228393555
	0.8744732727266786 0.2165360450744629
	0.8823939607190839 0.21657323837280273
	0.8840457628701927 0.2178800106048584
	0.8796877415937704 0.21748113632202148
	0.8768045314472548 0.21965289115905762
	0.876767087787496 0.2228717803955078
	0.8837648576052046 0.22686004638671875
	0.8750444773794261 0.22178196907043457
];
XGB_PCA_corrupted1 = [
	0.8822331817342322 0.07595491409301758
	0.8865103427532595 0.04269218444824219
	0.8668652395128374 0.04623699188232422
	0.8790459130360608 0.07125401496887207
	0.8880717695664785 0.03954291343688965
	0.8750178882195998 0.042459964752197266
	0.8800379741527035 0.0618748664855957
	0.8861418853255589 0.03818321228027344
	0.8701611644583905 0.041517019271850586
	0.8690085863315786 0.03901505470275879
];
XGB_PCA_corrupted2 = [
	0.8826920191153385 0.046305179595947266
	0.8913740872819363 0.04682207107543945
	0.8822835005898364 0.049082279205322266
	0.881080331819248 0.0467829704284668
	0.8846605314091208 0.048567771911621094
	0.8888942119478503 0.04706096649169922
	0.8782838748675026 0.04776191711425781
	0.88466472303207 0.04732513427734375
	0.8836506789114417 0.045037269592285156
	0.8762610511512595 0.04620695114135742
];  
XGB_PCA_corrupted3 = [
	0.8894331771170005 0.03234720230102539
	0.8783272457766537 0.03140377998352051
	0.8747563452357223 0.031203746795654297
	0.8984880472565201 0.031456947326660156
	0.8922753851675465 0.0325617790222168
	0.884642572811222 0.03225302696228027
	0.8903405960466968 0.03319501876831055
	0.8855587949465501 0.032450199127197266
	0.8838609648510336 0.03169679641723633
	0.8841451402211896 0.031999826431274414
];
XGB_PCA_small = [
	0.8923261167102152 0.021049022674560547
	0.8938563272353472 0.024121761322021484
	0.8756226319663655 0.021781206130981445
	0.8799437878431455 0.02324199676513672
	0.8744256903806162 0.023044824600219727
	0.8862254072350327 0.021010160446166992
	0.8701780125958934 0.019986867904663086
	0.8697353525453198 0.022307157516479492
	0.8740817857256598 0.023622989654541016
	0.8605198599740898 0.020044803619384766
];
XGB_PCA_corrupted1_small = [
	0.8838719138271209 0.02267289161682129
	0.8840017978008091 0.023073911666870117
	0.8828492271009172 0.022535085678100586
	0.8799850383350856 0.02126479148864746
	0.880423254239347 0.022164106369018555
	0.8746958959170722 0.024086713790893555
	0.8610867386357681 0.02339911460876465
	0.8693321041683088 0.021853923797607422
	0.8737078560376191 0.029547929763793945
	0.863116403428981 0.026027917861938477
];
XGB_PCA_corrupted2_small = [
	0.9019445777564479 0.037741899490356445
	0.8930252893613803 0.04265403747558594
	0.8748594421036945 0.036810874938964844
	0.8791734070342123 0.03751230239868164
	0.8725378496048299 0.03818917274475098
	0.8816770679077639 0.03730034828186035
	0.8724283941629438 0.038752079010009766
	0.8747753860846182 0.03726005554199219
	0.8709113451694983 0.03807187080383301
	0.8662917941508862 0.0370478630065918
];
XGB_PCA_corrupted3_small = [
	0.8870314083080042 0.024335145950317383
	0.8810948339926754 0.022773027420043945
	0.8806228803417645 0.02263498306274414
	0.8828067949594128 0.022642135620117188
	0.8796680054571364 0.022398948669433594
	0.8803416543262113 0.022938013076782227
	0.8669571295900926 0.02294611930847168
	0.8728265426535113 0.022744178771972656
	0.8648548922192973 0.022838592529296875
	0.8710769315582017 0.02262711524963379
];
XGB_LDA_balanced = [
	0.9002245182459602 0.028332948684692383
	0.9110504824482393 0.02861809730529785
	0.8956674746925657 0.02921319007873535
	0.8997545019498802 0.028582096099853516
	0.9055570371687359 0.02894425392150879
	0.9061078535189897 0.028352737426757812
	0.9104614499665576 0.028658390045166016
	0.8962795798842564 0.02842998504638672
	0.8985643464216517 0.028529882431030273
	0.9070575203734472 0.028927087783813477
];
XGB_PCA_balanced = [
	0.9261112902410035 0.032553911209106445
	0.9307268802523332 0.03189396858215332
	0.9257033617880654 0.031469106674194336
	0.9232091659980064 0.03449702262878418
	0.915897709426023 0.03141903877258301
	0.9208002086263849 0.031046152114868164
	0.9327293945955643 0.03122401237487793
	0.9224728103695135 0.03139519691467285
	0.9235565237729505 0.03152585029602051
	0.9171722446396076 0.031578779220581055
];
XGB_autoencoder_balanced = [
	0.8992406209695775 0.11548376083374023
	0.9070074303033752 0.1072547435760498
	0.8984715984690701 0.10743403434753418
	0.8977423977186934 0.10355997085571289
	0.9006609970268963 0.10830903053283691
	0.8994697412716728 0.10515403747558594
	0.9121039405155702 0.10317015647888184
	0.9043278211315817 0.10627412796020508
	0.892381293281372 0.10433006286621094
	0.8979658200807026 0.1033637523651123
];



% 4. MLP
MLP_original = [
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
];
MLP_balanced = [
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
    0., 0.
];
MLP_LDA = [
	0.9006761312484741 0.06679153442382812
	0.8982028365135193 0.06797027587890625
	0.9091496467590332 0.05902504920959473
	0.8986053466796875 0.06503701210021973
	0.8982307314872742 0.061986684799194336
	0.9024659991264343 0.05855512619018555
	0.9031493067741394 0.049271345138549805
	0.8996890187263489 0.04982280731201172
	0.9054163098335266 0.048635005950927734
	0.9029920101165771 0.0471196174621582
];
MLP_PCA = [
	0.9200280904769897 0.22349071502685547
	0.9220529794692993 0.24425911903381348
	0.9233958125114441 0.16495633125305176
	0.8662167191505432 0.1877291202545166
	0.9240403175354004 0.48507237434387207
	0.8600481152534485 0.14862561225891113
	0.9274644255638123 0.17676091194152832
	0.9224165678024292 0.3662688732147217
	0.9240151047706604 0.2849569320678711
	0.921552836894989 0.19961953163146973
];
MLP_autoencoder = [
	0.8984122276306152 0.35138893127441406
	0.8927643299102783 0.28829407691955566
	0.834872305393219 0.19993996620178223
	0.7886038422584534 0.38191771507263184
	0.9034028649330139 0.3258383274078369
	0.8815980553627014 0.39987921714782715
	0.8189547061920166 0.2929975986480713
	0.9052348732948303 0.39037013053894043
	0.8129526972770691 0.21025896072387695
	0.8703924417495728 0.27483367919921875
];
MLP_PCA_corrupted1 = [
	0.9111987948417664 0.12174654006958008
	0.9100891947746277 0.10872864723205566
	0.8802415728569031 0.09155941009521484
	0.8621614575386047 0.21872520446777344
	0.9135282039642334 0.09567975997924805
	0.8547802567481995 0.09013772010803223
	0.9138656854629517 0.10871386528015137
	0.9118204712867737 0.08608531951904297
	0.8061298727989197 0.09259891510009766
	0.9168928265571594 0.1130971908569336
];
MLP_PCA_corrupted2 = [
	0.8967518210411072 0.1812450885772705
	0.8479472398757935 0.14164257049560547
	0.8314461708068848 0.07907366752624512
	0.8536075949668884 0.09772682189941406
	0.895145058631897 0.09038615226745605
	0.887240469455719 0.06785845756530762
	0.8561191558837891 0.0690913200378418
	0.9007818102836609 0.11066389083862305
	0.9031550884246826 0.07551240921020508
	0.8938271403312683 0.06790518760681152
];  
MLP_PCA_corrupted3 = [
	0.8885656595230103 0.14117121696472168
	0.8900030255317688 0.15442323684692383
	0.833125114440918 0.15891718864440918
	0.8374212980270386 0.16337823867797852
	0.8400953412055969 0.22597718238830566
	0.8328811526298523 0.13115334510803223
	0.8989710807800293 0.13816118240356445
	0.8888571262359619 0.13984060287475586
	0.8889307379722595 0.2465953826904297
	0.8930947780609131 0.1385493278503418
];
MLP_PCA_small = [
	0.9223146438598633 0.05970597267150879
	0.8689941763877869 0.1594555377960205
	0.8641036748886108 0.0949714183807373
	0.8633298873901367 0.18008899688720703
	0.859068751335144 0.08612775802612305
	0.9145996570587158 0.049799442291259766
	0.8533990979194641 0.11293387413024902
	0.8559575080871582 0.09711217880249023
	0.9252470135688782 0.15954279899597168
	0.9170374274253845 0.09694933891296387
];
MLP_PCA_corrupted1_small = [
	0.9119448661804199 0.08722233772277832
	0.8734122514724731 0.11341714859008789
	0.9084001779556274 0.1069190502166748
	0.9084122180938721 0.0631873607635498
	0.8898261785507202 0.09024381637573242
	0.9089506268501282 0.07211494445800781
	0.8936029076576233 0.08030986785888672
	0.8912476301193237 0.10948014259338379
	0.8463995456695557 0.11043858528137207
	0.8353317975997925 0.07994222640991211
];
MLP_PCA_corrupted2_small = [
	0.9056840538978577 0.1458885669708252
	0.8504748940467834 0.10376977920532227
	0.8911032676696777 0.06400489807128906
	0.8440614938735962 0.08179593086242676
	0.8476095199584961 0.07901716232299805
	0.909044623374939 0.08208155632019043
	0.8654282689094543 0.07036662101745605
	0.9087180495262146 0.061986684799194336
	0.9013795256614685 0.06414437294006348
	0.8311223983764648 0.08740711212158203
];
MLP_PCA_corrupted3_small = [
	0.8934028744697571 0.09908795356750488
	0.8293280601501465 0.11214876174926758
	0.8291656970977783 0.08998703956604004
	0.8296611309051514 0.05719447135925293
	0.8268743753433228 0.06188154220581055
	0.8889842629432678 0.11595344543457031
	0.837558925151825 0.11815261840820312
	0.9079921841621399 0.10587263107299805
	0.8985119462013245 0.08280420303344727
	0.8869867324829102 0.14167284965515137
];
MLP_LDA_balanced = [
	0.9050910472869873 0.21216225624084473
	0.9180908799171448 0.19592595100402832
	0.9065797924995422 0.15715551376342773
	0.5 0.040222883224487305
	0.9093778133392334 0.15270018577575684
	0.5006369352340698 0.0443267822265625
	0.5003095269203186 0.03827023506164551
	0.5 0.03831291198730469
	0.9092466235160828 0.13969969749450684
	0.9093634486198425 0.1236879825592041
];
MLP_PCA_balanced = [
	0.9276329874992371 0.4250671863555908
	0.9329416155815125 0.31707310676574707
	0.9275830388069153 0.44024085998535156
	0.9263243079185486 0.3030052185058594
	0.9194087386131287 0.29706263542175293
	0.9226197600364685 0.3029952049255371
	0.9352530837059021 0.29024839401245117
	0.9261877536773682 0.33993053436279297
	0.9245173931121826 0.28705453872680664
	0.9149109721183777 0.33452606201171875
];
MLP_autoencoder_balanced = [
	0.9122980237007141 0.5961730480194092
	0.9094544649124146 0.5268595218658447
	0.9169381856918335 0.7233848571777344
	0.9111201167106628 0.4546992778778076
	0.9110220074653625 0.518876314163208
	0.9064294695854187 0.6220169067382812
	0.8868830800056458 0.34967565536499023
	0.9134688377380371 0.5697183609008789
	0.9129975438117981 0.5825881958007812
	0.9064039587974548 0.41913580894470215
];


%% Plot accuracies 
figure()
tl = tiledlayout(5, 3);
title(tl,'AUC vs. classification algorithm for each dimensionality reduction method', ...
    'Fontsize', 20)
AUC_eSPA = {eSPA_original(:,1); eSPA_balanced(:,1); eSPA_LDA(:,1); 
    eSPA_PCA(:,1); eSPA_autoencoder(:,1); eSPA_PCA_corrupted1(:,1);
    eSPA_PCA_corrupted2(:,1); eSPA_PCA_corrupted3(:,1); eSPA_PCA_small(:,1);
    eSPA_PCA_corrupted1_small(:,1); eSPA_PCA_corrupted2_small(:,1); eSPA_PCA_corrupted3_small(:,1);
    eSPA_LDA_balanced(:,1); eSPA_PCA_balanced(:,1); eSPA_autoencoder_balanced(:,1);
};
AUC_RF = {RF_original(:,1); RF_balanced(:,1); RF_LDA(:,1); 
    RF_PCA(:,1); RF_autoencoder(:,1); RF_PCA_corrupted1(:,1);
    RF_PCA_corrupted2(:,1); RF_PCA_corrupted3(:,1); RF_PCA_small(:,1);
    RF_PCA_corrupted1_small(:,1); RF_PCA_corrupted2_small(:,1); RF_PCA_corrupted3_small(:,1);
    RF_LDA_balanced(:,1); RF_PCA_balanced(:,1); RF_autoencoder_balanced(:,1);
};
AUC_XGB = {XGB_original(:,1); XGB_balanced(:,1); XGB_LDA(:,1); 
    XGB_PCA(:,1); XGB_autoencoder(:,1); XGB_PCA_corrupted1(:,1);
    XGB_PCA_corrupted2(:,1); XGB_PCA_corrupted3(:,1); XGB_PCA_small(:,1);
    XGB_PCA_corrupted1_small(:,1); XGB_PCA_corrupted2_small(:,1); XGB_PCA_corrupted3_small(:,1);
    XGB_LDA_balanced(:,1); XGB_PCA_balanced(:,1); XGB_autoencoder_balanced(:,1);
};
AUC_MLP = {MLP_original(:,1); MLP_balanced(:,1); MLP_LDA(:,1); 
    MLP_PCA(:,1); MLP_autoencoder(:,1); MLP_PCA_corrupted1(:,1);
    MLP_PCA_corrupted2(:,1); MLP_PCA_corrupted3(:,1); MLP_PCA_small(:,1);
    MLP_PCA_corrupted1_small(:,1); MLP_PCA_corrupted2_small(:,1); MLP_PCA_corrupted3_small(:,1);
    MLP_LDA_balanced(:,1); MLP_PCA_balanced(:,1); MLP_autoencoder_balanced(:,1);
};

for i = 1 : 5*3
    nexttile
    h = boxplot([AUC_eSPA{i} AUC_RF{i} AUC_XGB{i} AUC_MLP{i}], 'Notch', 'on');
    set(h,{'linew'},{1})
    set(gcf,'Position',[10 100 800  600]);
    ylabel('AUC on validation data')
    grid on
    title(data_reduction{i})
    xticklabels({'eSPA', 'RF', 'XGB', 'MLP'})
    %ylim([.8, .95])
end        

%% Plot running times  
figure()
tl2 = tiledlayout(5, 3);
title(tl2,'Running time vs. classif. algorithm for each dimensionality reduction method', ...
    'Fontsize', 20)
time_eSPA = {eSPA_original(:,2); eSPA_balanced(:,2); eSPA_LDA(:,2); 
    eSPA_PCA(:,2); eSPA_autoencoder(:,2); eSPA_PCA_corrupted1(:,2);
    eSPA_PCA_corrupted2(:,2); eSPA_PCA_corrupted3(:,2); eSPA_PCA_small(:,2);
    eSPA_PCA_corrupted1_small(:,2); eSPA_PCA_corrupted2_small(:,2); eSPA_PCA_corrupted3_small(:,2);
    eSPA_LDA_balanced(:,2); eSPA_PCA_balanced(:,2); eSPA_autoencoder_balanced(:,2);
};
time_RF = {RF_original(:,2); RF_balanced(:,2); RF_LDA(:,2); 
    RF_PCA(:,2); RF_autoencoder(:,2); RF_PCA_corrupted1(:,2);
    RF_PCA_corrupted2(:,2); RF_PCA_corrupted3(:,2); RF_PCA_small(:,2);
    RF_PCA_corrupted1_small(:,2); RF_PCA_corrupted2_small(:,2); RF_PCA_corrupted3_small(:,2);
    RF_LDA_balanced(:,2); RF_PCA_balanced(:,2); RF_autoencoder_balanced(:,2);
};
time_XGB = {XGB_original(:,2); XGB_balanced(:,2); XGB_LDA(:,2); 
    XGB_PCA(:,2); XGB_autoencoder(:,2); XGB_PCA_corrupted1(:,2);
    XGB_PCA_corrupted2(:,2); XGB_PCA_corrupted3(:,2); XGB_PCA_small(:,2);
    XGB_PCA_corrupted1_small(:,2); XGB_PCA_corrupted2_small(:,2); XGB_PCA_corrupted3_small(:,2);
    XGB_LDA_balanced(:,2); XGB_PCA_balanced(:,2); XGB_autoencoder_balanced(:,2);
};
time_MLP = {MLP_original(:,2); MLP_balanced(:,2); MLP_LDA(:,2); 
    MLP_PCA(:,2); MLP_autoencoder(:,2); MLP_PCA_corrupted1(:,2);
    MLP_PCA_corrupted2(:,2); MLP_PCA_corrupted3(:,2); MLP_PCA_small(:,2);
    MLP_PCA_corrupted1_small(:,2); MLP_PCA_corrupted2_small(:,2); MLP_PCA_corrupted3_small(:,2);
    MLP_LDA_balanced(:,2); MLP_PCA_balanced(:,2); MLP_autoencoder_balanced(:,2);
};
for i = 1 : 5*3
    nexttile
    h = boxplot([time_eSPA{i} time_RF{i} time_XGB{i} time_MLP{i}], 'Notch', 'on');
    set(h,{'linew'},{1})
    set(gcf,'Position',[10 100 800  600]);
    ylabel('CPU time (sec.)')
    set(gca,'YScale','log');
    grid on
    title(data_reduction{i})
    xticklabels({'eSPA', 'RF', 'XGB', 'MLP'})
    %ylim([.03, 6])
end
       
%% Write [averages, running times] for eSPA
average_original = mean(eSPA_original, 1)
average_balanced = mean(eSPA_balanced, 1)
average_LDA = mean(eSPA_LDA, 1)
average_PCA = mean(eSPA_PCA, 1)
average_autoencoder = mean(eSPA_autoencoder, 1)
average_PCA_corrupted1 = mean(eSPA_PCA_corrupted1, 1)
average_PCA_corrupted2 = mean(eSPA_PCA_corrupted2, 1)
average_PCA_corrupted3 = mean(eSPA_PCA_corrupted3, 1)
average_PCA_small = mean(eSPA_PCA_small, 1)
average_PCA_corrupted1_small = mean(eSPA_PCA_corrupted1_small, 1)
average_PCA_corrupted2_small = mean(eSPA_PCA_corrupted2_small, 1)
average_PCA_corrupted3_small = mean(eSPA_PCA_corrupted3_small, 1)
average_LDA_balanced = mean(eSPA_LDA_balanced, 1)
average_PCA_balanced = mean(eSPA_PCA_balanced, 1)
average_autoencoder_balanced = mean(eSPA_autoencoder_balanced, 1)

        
        
        
        
        
        
        
        