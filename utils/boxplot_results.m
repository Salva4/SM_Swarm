% Plot accuracies & running times

close all
clear all
clc

%% Table & labels
% Labels
data_reduction = {'LDA'; 'PCA-10'; 'Autoencoder'; 'PCA-10 1%-corrupted'; 
    'PCA-10 2%-corrupted'; 'PCA-10 3%-corrupted'; 
    'PCA-10 1%-corrupted SMALL'; 'PCA-10 2%-corrupted SMALL'; 
    'PCA-10 3%-corrupted SMALL'};

% Table
% 1. eSPA
eSPA_LDA = [
    0.929858429858430, 0.043665647468393
    0.930930930930931, 0.0510595713067857
    0.932646932646933, 0.044828001106071
    0.930287430287430, 0.046739617682500
    0.932432432432432, 0.065422168693571
    0.923208923208923, 0.044731733136607
    0.931788931788932, 0.039846749757143
    0.925353925353925, 0.044869976625536
    0.930072930072930, 0.053450217861071
    0.930716430716431, 0.042459186302679
];
eSPA_PCA10 = [
    0.902616902616903, 0.127563600401607
    0.904118404118404, 0.121505144264107
    0.904118404118404, 0.124499246358393
    0.904761904761905, 0.119356745730357
    0.906263406263406, 0.124461641146607
    0.894036894036894, 0.125585241614107
    0.902831402831403, 0.115650909181607
    0.905190905190905, 0.119502622750179
    0.912483912483912, 0.116274479765714
    0.901758901758902, 0.117716295675357
];
eSPA_PCA10_corrupted1 = [
    0.895109395109395, 0.124478656271429
    0.896825396825397, 0.121235677223036
    0.891033891033891, 0.120698137731071
    0.897039897039897, 0.123114694772143
    0.901758901758902, 0.119111910027143
    0.894251394251394, 0.122253643349464
    0.894894894894895, 0.119060744018214
    0.897468897468897, 0.118698390985000
    0.903045903045903, 0.116081134545357
    0.893822393822394, 0.116457981595357
];
eSPA_PCA10_corrupted2 = [
    0.883740883740884, 0.123223708100357
    0.887816387816388, 0.126574842939464
    0.883526383526384, 0.121123242956786
    0.888245388245388, 0.124274562746786
    0.894036894036894, 0.118599761280179
    0.887601887601888, 0.123496646387321
    0.890175890175890, 0.120437629434821
    0.886743886743887, 0.118157622453393
    0.897683397683398, 0.123682942903750
    0.887172887172887, 0.118210071213393
];  
eSPA_PCA10_corrupted3 = [
    0.878378378378378, 0.121266318967500
    0.881381381381381, 0.119385858945357
    0.880737880737881, 0.120788639503750
    0.881381381381381, 0.124085250632500
    0.886314886314886, 0.120995326076071
    0.876447876447876, 0.122067234939464
    0.882239382239382, 0.118039596271786
    0.880308880308880, 0.121600085797143
    0.885027885027885, 0.123312172262143
    0.874946374946375, 0.116917991913750
];
eSPA_PCA10_corrupted1_SMALL = [
    0.897000000000000, 0.043685000815000
    0.887500000000000, 0.043987143380357
    0.893000000000000, 0.044751887529821
    0.891500000000000, 0.042805497821964
    0.887500000000000, 0.041912110252857
    0.889000000000000, 0.042449941359286
    0.896500000000000, 0.041795019480893
    0.895500000000000, 0.041631737838214
    0.889000000000000, 0.041879297757321
    0.883000000000000, 0.042149346052679
];
eSPA_PCA10_corrupted2_SMALL = [
    0.896000000000000, 0.041874683914464
    0.879000000000000, 0.043449938003393
    0.881500000000000, 0.042155930734286
    0.880500000000000, 0.044178435893929
    0.882500000000000, 0.042355162303750
    0.887500000000000, 0.042729324706429
    0.892000000000000, 0.043638256044286
    0.891500000000000, 0.041849218875357
    0.880500000000000, 0.041429357815357
    0.876000000000000, 0.042445805663214
];
eSPA_PCA10_corrupted3_SMALL = [
    0.879500000000000, 0.042411730815000
    0.869000000000000, 0.044666222000893
    0.862500000000000, 0.042850863102679
    0.876500000000000, 0.043808882626429
    0.876500000000000, 0.042497642000357
    0.868500000000000, 0.043642070769821
    0.885500000000000, 0.043356650514464
    0.888000000000000, 0.043206185537857
    0.875000000000000, 0.042638977049464
    0.853500000000000, 0.043431323582500
];
eSPA_autoencoder = [
    0.849849849849850, 0.175236894403571
    0.848991848991849, 0.165330371153393
    0.847919347919348, 0.172298872325357
    0.843843843843844, 0.166926446897857
    0.838481338481338, 0.159511822225536
    0.845988845988846, 0.173700086927321
    0.851994851994852, 0.174479572308036
    0.852423852423852, 0.161527273751964
    0.850922350922351, 0.161021535277143
    0.841269841269841, 0.161916348336071
];

% 2. Random Forest
RF_LDA = [
    0.8681520685197155, 0.9797878265380859
    0.8691635065655876, 0.9152648448944092
    0.8662629445422967, 0.9083619117736816
    0.8724585759068518, 0.9827561378479004
    0.8807314209204034, 0.9269859790802002
    0.8693921459612923, 0.9181389808654785
    0.8703137856586116, 0.8994958400726318
    0.8717201166180759, 0.8767280578613281
    0.8732357748954731, 0.9068269729614258
    0.8670221826194222, 0.8750867843627930
];
RF_PCA10 = [
    0.8503569697109612, 2.5950610637664795
    0.8489939744821146, 2.5469148159027100
    0.8462123836257462, 2.5030801296234130
    0.8543023478984070, 2.5693051815032960
    0.8567243100088662, 2.6040618419647217
    0.8496135322119545, 2.5558247566223145
    0.8486615774168033, 2.6055080890655518
    0.8538516358924522, 2.5594921112060547
    0.8562714085050804, 2.5493819713592530
    0.8476021615854861, 2.4906620979309080
];
RF_PCA10_corrupted1 = [
    0.8481161695447409,  4.5831232070922852
    0.8506037981036069,  4.1753919124603271
    0.8464294875774875,  4.0041692256927490
    0.8525835412534921,  4.0583009719848633
    0.8513298748420571,  4.1111459732055664
    0.8505245977412319,  4.2549591064453125
    0.8469053090063525,  4.3196198940277100
    0.8511305474570781,  4.2357759475708008
    0.8548593712778977,  4.4077677726745605
    0.8463052709104301,  4.2946708202362061
];
RF_PCA10_corrupted2 = [
    0.8423590590082186,  4.8642220497131348
    0.8452901338376201,  4.9008259773254395
    0.8417375505282386,  4.8152189254760742
    0.8472117373595206,  4.8658061027526855
    0.8490527115369173,  5.0352108478546143
    0.8448071526560099,  4.9729871749877930
    0.8451273493434057,  5.1581439971923828
    0.8484029802397148,  4.8016247749328613
    0.8490430346889657,  4.8347570896148682
    0.8448365323707427,  4.9806280136108398
];
RF_PCA10_corrupted3 = [
    0.8368905254409456,  5.2784361839294434
    0.8456015048838934,  5.0675830841064453
    0.8407853975432737,  4.5614569187164307
    0.8533277887957691,  4.5764369964599609
    0.8476887898667462,  4.5187461376190186
    0.8506730494027390,  4.8343720436096191
    0.8412706446495634,  4.5613601207733154
    0.8485195983155168,  4.6782851219177246
    0.8517609393867207,  4.6932890415191650
    0.8462852674552878,  4.7119801044464111
];
RF_PCA10_corrupted1_SMALL = [
    0.8756910005510230,  1.4050209522247314
    0.8713037170866728,  1.3008058071136475
    0.8720437682612368,  1.2971072196960449
    0.8680452506747800,  1.3565382957458496
    0.8650325999790803,  1.3224811553955078
    0.8651496720964671,  1.3854730129241943
    0.8792352827721224,  1.3776872158050537
    0.8738712186897055,  1.3224592208862305
    0.8750904624318717,  1.2916548252105713
    0.8527247167783013,  1.3057301044464111
];
RF_PCA10_corrupted2_SMALL = [
    0.8777395616701328,  1.5281791687011719
    0.8713293045981870,  1.4612116813659668
    0.8619145678945443,  1.5855741500854492
    0.8616726071091321,  1.7061939239501953
    0.8665329750728538,  1.5166451930999756
    0.8704648825893802,  1.5083570480346680
    0.8748170206219014,  1.4706161022186279
    0.8726951727856466,  1.5000197887420654
    0.8712413477282537,  1.4176940917968750
    0.8528184349072466,  1.5464291572570801
];
RF_PCA10_corrupted3_SMALL = [
    0.8667746493894312,  1.6359639167785645
    0.8712013670406152,  1.5959868431091309
    0.8648882988091529,  1.5938711166381836
    0.8653751174802863,  1.5426797866821289
    0.8672820379007796,  1.5331141948699951
    0.8603589309639659,  1.5543749332427979
    0.8704400089636204,  1.6029162406921387
    0.8696713811314030,  1.5752091407775879
    0.8706977495033490,  1.5148258209228516
    0.8554618374266104,  1.6234710216522217
];
RF_autoencoder = [
    0.8828839227998893,  4.9482429027557373
    0.8814862390199643,  4.8335909843444824
    0.8741199426829228,  4.8013539314270020
    0.8887232636001109,  4.8130300045013428
    0.8843979078613347,  4.9627840518951416
    0.8850276835619187,  4.7182719707489014
    0.8812910055052399,  5.1110918521881104
    0.8782183349530288,  4.7493932247161865
    0.8879348442159045,  4.8536279201507568
    0.8771851501623007,  5.3161840438842773
];


% 3. XGBoost
XGB_LDA = [
    0.9012085603472159,  0.3193159103393555
    0.9069399392462951,  0.3078110218048096
    0.8980493730740449,  0.3056781291961670
    0.9030473631458853,  0.3250880241394043
    0.9057189657630669,  0.3248310089111328
    0.8983178082777633,  0.3107070922851562
    0.9006598479009374,  0.3037083148956299
    0.9030126336248786,  0.3053791522979736
    0.9034411733298287,  0.3227021694183350
    0.8987370545821551,  0.3007040023803711
];
XGB_PCA10 = [
    0.8517810970542063,  0.8244938850402832
    0.8534964953620462,  0.6548590660095215
    0.8494086929811089,  0.6638071537017822
    0.8545731772332757,  0.6968841552734375
    0.8567347575915004,  0.6540670394897461
    0.8534477795579184,  0.6755809783935547
    0.8507533371991965,  0.6614761352539062
    0.8505539358600583,  0.6789929866790771
    0.8527984047840051,  0.6641259193420410
    0.8517674264948795,  0.6694240570068359
];
XGB_PCA10_corrupted1 = [
    0.8794037398460691,  0.3368740081787109
    0.8721903799856711,  0.3065919876098633
    0.8698095186484766,  0.3060860633850098
    0.8661321197512930,  0.3145959377288818
    0.8744060077988013,  0.3379137516021729
    0.8671638107326706,  0.3349609375000000
    0.8771086526808918,  0.3287742137908936
    0.8776341086440231,  0.3198759555816650
    0.8659201251044643,  0.3079731464385986
    0.8563218390804598,  0.3134257793426514
];
XGB_PCA10_corrupted2 = [
    0.8834097655486234,  0.3452160358428955
    0.8733028804862963,  0.4424397945404053
    0.8739438400643068,  0.4739062786102295
    0.8683000003344635,  0.3482301235198975
    0.8706601312996914,  0.3109788894653320
    0.8696539383682391,  0.3114118576049805
    0.8781822803494921,  0.3109371662139893
    0.8802883511482298,  0.3330080509185791
    0.8680143509931375,  0.5091838836669922
    0.8580501116348300,  0.3193709850311279
];
XGB_PCA10_corrupted3 = [
    0.8723582004657032,  0.3529889583587646
    0.8746923936115771,  0.3085551261901855
    0.8701753078725970,  0.3101770877838135
    0.8709288830370173,  0.3167481422424316
    0.8740303516658775,  0.3078489303588867
    0.8712802341160709,  0.3107750415802002
    0.8793796594939123,  0.3101232051849365
    0.8774656125091531,  0.3117690086364746
    0.8785546656651281,  0.3086550235748291
    0.8545467074616170,  0.3102450370788574
];
XGB_PCA10_corrupted1_SMALL = [
    0.8774173909952185,  1.6482207775115967
    0.8673610153124569,  1.6995580196380615
    0.8780860643337443,  1.6198101043701172
    0.8617138576010720,  1.6633381843566895
    0.8684061945021488,  1.7161378860473633
    0.8641932867921867,  1.6432299613952637
    0.8771705284188017,  1.6440000534057617
    0.8735016695396686,  1.6952891349792480
    0.8719655962723438,  1.6903967857360840
    0.8537804239367128,  1.7067692279815674
];
XGB_PCA10_corrupted2_SMALL = [
    0.8785860928918040,  1.9111249446868896
    0.8714572421557590,  2.0027730464935303
    0.8694437294243561,  1.9572539329528809
    0.8647006161931593,  1.8755187988281250
    0.8639090057371914,  1.9296109676361084
    0.8662514984838868,  1.8226258754730225
    0.8701027583200570,  1.8349177837371826
    0.8757526636668643,  1.8878998756408691
    0.8699279265626252,  1.8027670383453369
    0.8514815733620001,  1.8149559497833252
];
XGB_PCA10_corrupted3_SMALL = [
    0.8855738637373576,  1.9228329658508301
    0.8724162175872980,  1.9535448551177979
    0.8709226920280529,  1.9368169307708740
    0.8628699862535523,  1.9362299442291260
    0.8687829753540334,  1.9536063671112061
    0.8674987659544460,  1.8904061317443848
    0.8782647813333718,  1.9080529212951660
    0.8749461669128423,  1.9174351692199707
    0.8769584636047261,  1.9072828292846680
    0.8580969706993027,  1.9718899726867676
];
XGB_autoencoder = [
    0.8895284075168530,  1.3993198871612549
    0.8931505016833562,  1.4709739685058594
    0.8783791014878258,  1.3404233455657959
    0.8974217900326275,  1.3224210739135742
    0.8885283903839629,  1.3674311637878418
    0.8831233078155482,  1.3914229869842529
    0.8828708517287205,  1.5266067981719971
    0.8852931648850016,  1.5201530456542969
    0.8881632016034304,  1.5083739757537842
    0.8836741497773857,  1.4982919692993164
];


% 4. MLP
MLP_LDA = [
    0.9266841351925386,  0.5811965465545654
    0.9287753791290418,  0.6475965976715088
    0.9356679939349775,  1.5608389377593994
    0.9291855927323908,  0.3696830272674561
    0.9301736838913844,  0.3617131710052490
    0.9303737034124966,  0.5788865089416504
    0.9329675224106790,  0.4952185153961182
    0.9287463556851312,  0.3669998645782471
    0.9387340950722447,  0.5085618495941162
    0.9337000633442746,  0.5374014377593994
];
MLP_PCA10 = [
    0.8951729730353679,  0.6424255371093750
    0.8889250781854982,  0.4984853267669678
    0.8855322988592136,  0.9332618713378906
    0.8827010427995650,  0.7588827610015869
    0.8933185045944781,  0.8594336509704590
    0.8795604679393758,  0.6985759735107422
    0.8963020752744304,  0.8939256668090820
    0.8999546485260771,  0.5168788433074951
    0.9024458061919352,  0.6934165954589844
    0.8879263994083826,  0.8592710494995117
];
MLP_PCA10_corrupted1 = [
    0.8823708587986232,  0.9134705066680908
    0.8879373012267150,  0.6826930046081543
    0.8789815936724348,  0.9725332260131836
    0.8712368680900660,  0.7223036289215088
    0.8771368430179013,  0.5677051544189453
    0.8733564268867925,  1.0208299160003662
    0.8870837830934155,  0.5060131549835205
    0.8808924109517169,  0.5311489105224609
    0.8725926974559951,  0.7786107063293457
    0.8796962103722058,  1.3661699295043945
];
MLP_PCA10_corrupted2 = [
    0.8664541402649886,  0.8864796161651611
    0.8834189288031715,  0.6548628807067871
    0.8624501690480554,  1.6069180965423584
    0.8640343020754729,  1.9404284954071045
    0.8739319163316583,  1.7282965183258057
    0.8660105721342076,  0.8745565414428711
    0.8777059155713420,  0.4961824417114258
    0.8822143497418674,  0.5437834262847900
    0.8683757043831201,  0.9029304981231689
    0.8722055388722055,  0.7631475925445557
];
MLP_PCA10_corrupted3 = [
    0.8691647954136338,  0.8754296302795410
    0.8736013848903820,  0.5243473052978516
    0.8593979168018692,  1.2965874671936035
    0.8625290300871378,  0.6454253196716309
    0.8603802659677733,  0.8553278446197510
    0.8535418940443107,  1.1258668899536133
    0.8703655755966525,  0.5417110919952393
    0.8541315942826618,  0.5343008041381836
    0.8553936276088167,  0.8540656566619873
    0.8676402767409406,  0.6855764389038086
];
MLP_PCA10_corrupted1_SMALL = [
    0.8858024998040299,  0.6155748367309570
    0.8844566553059287,  0.5601639747619629
    0.8805704099821748,  0.3429226875305176
    0.8893175321251146,  0.2606182098388672
    0.8595556634478790,  0.3027217388153076
    0.8661177749719415,  0.2934157848358154
    0.8674666761202495,  0.2563202381134033
    0.8872130490979594,  0.2000269889831543
    0.8836646775412347,  0.3935437202453613
    0.8659212727009337,  0.8212146759033203
];
MLP_PCA10_corrupted2_SMALL = [
    0.8760645465237717,  0.3460807800292969
    0.8697848525698563,  0.3581519126892090
    0.8762718753604136,  0.2096879482269287
    0.8675110347965579,  0.2699458599090576
    0.8581013838869326,  0.3289263248443604
    0.8669760946356690,  0.4068441390991211
    0.8640013471355644,  0.2480962276458740
    0.8874491869918699,  0.2440526485443115
    0.8738401676144867,  0.3178298473358154
    0.8557156894857517,  0.3496859073638916
];
MLP_PCA10_corrupted3_SMALL = [
    0.8605617385029556,  0.6808302402496338
    0.8620790584465601,  0.7367455959320068
    0.8537154145314840,  0.3856942653656006
    0.8536945829058532,  0.3900368213653564
    0.8392279457509021,  0.3214499950408936
    0.8674192503207219,  0.2021353244781494
    0.8631152477837812,  0.2496428489685059
    0.8841735627177700,  0.2672042846679688
    0.8666615574767559,  0.3318581581115723
    0.8472132359737499,  0.3685951232910156
];
MLP_autoencoder = [
    0.8629028534490720,  2.0170054435729980
    0.8564076796272355,  1.5268301963806152
    0.8500634768656643,  1.0335650444030762
    0.8673448062610624,  1.4332187175750732
    0.8598790702552815,  1.1323544979095459
    0.8677192890111228,  1.2282826900482178
    0.8533324271210068,  1.4388396739959717
    0.8454292193067703,  2.4592676162719727
    0.8557664347497514,  1.3913600444793701
    0.8671515989125395,  1.6280009746551514
];

%% Plot accuracies 
figure()
tl = tiledlayout(3, 3);
title(tl,'AUC vs. classification algorithm for each dimensionality reduction method', ...
    'Fontsize', 20)
AUC_eSPA = {eSPA_LDA(:,1); eSPA_PCA10(:,1); eSPA_autoencoder(:,1); eSPA_PCA10_corrupted1(:,1);
    eSPA_PCA10_corrupted1(:,1); eSPA_PCA10_corrupted3(:,1);
    eSPA_PCA10_corrupted1_SMALL(:,1); eSPA_PCA10_corrupted1_SMALL(:,1); 
    eSPA_PCA10_corrupted3_SMALL(:,1)};
AUC_RF = {RF_LDA(:,1); RF_PCA10(:,1); RF_autoencoder(:,1); RF_PCA10_corrupted1(:,1);
    RF_PCA10_corrupted1(:,1); RF_PCA10_corrupted3(:,1);
    RF_PCA10_corrupted1_SMALL(:,1); RF_PCA10_corrupted1_SMALL(:,1); 
    RF_PCA10_corrupted3_SMALL(:,1)};
AUC_XGB = {XGB_LDA(:,1); XGB_PCA10(:,1); XGB_autoencoder(:,1); XGB_PCA10_corrupted1(:,1);
    XGB_PCA10_corrupted1(:,1); XGB_PCA10_corrupted3(:,1);
    XGB_PCA10_corrupted1_SMALL(:,1); XGB_PCA10_corrupted1_SMALL(:,1); 
    XGB_PCA10_corrupted3_SMALL(:,1)};
AUC_MLP = {MLP_LDA(:,1); MLP_PCA10(:,1); MLP_autoencoder(:,1); MLP_PCA10_corrupted1(:,1);
    MLP_PCA10_corrupted1(:,1); MLP_PCA10_corrupted3(:,1);
    MLP_PCA10_corrupted1_SMALL(:,1); MLP_PCA10_corrupted1_SMALL(:,1); 
    MLP_PCA10_corrupted3_SMALL(:,1)};
for i = 1 : 9
    nexttile
    h = boxplot([AUC_eSPA{i} AUC_RF{i} AUC_XGB{i} AUC_MLP{i}], 'Notch', 'on');
    set(h,{'linew'},{1})
    set(gcf,'Position',[10 100 800  600]);
    ylabel('AUC on validation data')
    grid on
    title(data_reduction{i})
    xticklabels({'eSPA', 'RF', 'XGB', 'MLP'})
    ylim([.8, .95])
end        

%% Plot running times  
figure()
tl2 = tiledlayout(3, 3);
title(tl2,'Running time vs. classif. algorithm for each dimensionality reduction method', ...
    'Fontsize', 20)
time_eSPA = {eSPA_LDA(:,2); eSPA_PCA10(:,2); eSPA_autoencoder(:,2); eSPA_PCA10_corrupted1(:,2);
    eSPA_PCA10_corrupted2(:,2); eSPA_PCA10_corrupted3(:,2);
    eSPA_PCA10_corrupted1_SMALL(:,2); eSPA_PCA10_corrupted2_SMALL(:,2); 
    eSPA_PCA10_corrupted3_SMALL(:,2)};
time_RF = {RF_LDA(:,2); RF_PCA10(:,2); RF_autoencoder(:,2); RF_PCA10_corrupted1(:,2);
    RF_PCA10_corrupted2(:,2); RF_PCA10_corrupted3(:,2);
    RF_PCA10_corrupted1_SMALL(:,2); RF_PCA10_corrupted2_SMALL(:,2); 
    RF_PCA10_corrupted3_SMALL(:,2)};
time_XGB = {XGB_LDA(:,2); XGB_PCA10(:,2); XGB_autoencoder(:,2); XGB_PCA10_corrupted1(:,2);
    XGB_PCA10_corrupted2(:,2); XGB_PCA10_corrupted3(:,2);
    XGB_PCA10_corrupted1_SMALL(:,2); XGB_PCA10_corrupted2_SMALL(:,2); 
    XGB_PCA10_corrupted3_SMALL(:,2)};
time_MLP = {MLP_LDA(:,2); MLP_PCA10(:,2); MLP_autoencoder(:,2); MLP_PCA10_corrupted1(:,2);
    MLP_PCA10_corrupted2(:,2); MLP_PCA10_corrupted3(:,2);
    MLP_PCA10_corrupted1_SMALL(:,2); MLP_PCA10_corrupted2_SMALL(:,2); 
    MLP_PCA10_corrupted3_SMALL(:,2)};
for i = 1 : 9
    nexttile
    h = boxplot([time_eSPA{i} time_RF{i} time_XGB{i} time_MLP{i}], 'Notch', 'on');
    set(h,{'linew'},{1})
    set(gcf,'Position',[10 100 800  600]);
    ylabel('CPU time (sec.)')
    set(gca,'YScale','log');
    grid on
    title(data_reduction{i})
    xticklabels({'eSPA', 'RF', 'XGB', 'MLP'})
    ylim([.03, 6])
end
       
        
        
        
        
        
        
        
        
        