import numpy as np
import matplotlib.pyplot as plt



returns = []
path = './data/push-v2'
for i in range(50):
    for j in range(45):
        file_name = path+'/goal_idx%d'%i+'/trj_evalsample%d_step49500.npy'%j
        traj = np.load(file_name,allow_pickle=1)
        returns.append(sum(s[2] for s in traj))


average_return = np.mean(returns)
std_return = np.std(returns)
return_plots = np.zeros(500)
for r in returns:
 idx = int(np.floor(r/10))
 # print(idx)
 return_plots[idx]+=1
return_plots /=len(returns)


array = [ 21.207345962524414, 78.89094500168149 ,
 74.24502563476562, 218.9976900120081 ,
 23.134618759155273, 22.646142573940537 ,
 61.630653381347656, 32.14097330820273 ,
 51.13279342651367, 32.61684679733313 ,
 83.41130065917969, 114.6340807835531 ,
 13.633193016052246, 73.68637744429203 ,
 78.64085388183594, 70.9905121998283 ,
 37.1361083984375 ,85.15271261807749 ,
 8.857566833496094, 69.5725090024118 ,
 5.744174480438232, 4312.3654450609865 ,
 38.052818298339844 ,157.69540430477508 ,
 186.90296936035156, 342.45991357616947 ,
 16.013139724731445, 42.68578938730728 ,
 56.48713684082031, 56.21794924350344 ,
 31.564849853515625, 2856.0538007263413 ,
 76.78848266601562, 214.4939963278924 ,
 41.25037384033203, 2249.4575922607632 ,
 5.929319381713867, 4305.640553488969 ,
 73.43765258789062, 209.17905352312403 ,
 0.25711578130722046, 4670.252298425437 ,
 21.948062896728516, 2429.153641653883 ,
 36.35655212402344, 49.46675981590657 ,
 0.5662634968757629, 4436.890368548283 ,
 48.970306396484375, 96.20415967705397 ,
 0.2430468648672104, 4667.8810201579845 ,
 0.26221805810928345, 4671.704431199351 ,
 0.2644940912723541, 4671.686192772824 ,
 46.181488037109375, 2140.839994443226 ,
 0.2648427486419678, 4671.6457651891005 ,
 8.014219284057617, 20.07813616472144 ,
 34.183197021484375, 153.84550773994062 ,
 0.08948919177055359, 4543.62021755157 ,
 17.22488021850586, 50.400832429316985 ,
 3.9885377883911133, 23.41614775075459 ,
 9.707230567932129, 260.5886933889377 ,
 0.0888444259762764, 4542.157030430271 ,
 0.14837679266929626, 4536.013861066674 ,
 0.08580988645553589, 4544.434085634371 ,
 18.33502197265625, 43.247473388515914 ,
 93.48113250732422, 101.06679259074048 ,
 54.17123031616211, 226.65213360552855 ,
 10.357330322265625, 105.08967090877654 ,
 98.14056396484375, 224.25010865157307 ,
 7.224633693695068, 22.839849734136457 ,
 8.21849536895752 ,23.226191010657615 ,
 7.588539123535156, 24.85764877124374 ,
 7.600669860839844, 24.89462521752572 ,
 7.600281715393066, 24.892706745204897 ,
 7.582900524139404, 24.875237584291234 ,
 2.165472984313965, 4195.572881221507 ,
 58.3193359375 ,1209.180801383145 ,
 12.889384269714355, 44.42857188645455 ,
 7.680802345275879, 18.52814346720536 ,
 0.391788512468338, 4300.619320592326 ,
 0.09144315868616104, 4579.5295446840555 ,
 0.4315984547138214, 4589.107490817538 ,
 0.07079341262578964, 4600.327375262688 ,
 67.0701904296875 ,1077.2222612103178 ,
 0.0693230926990509, 4597.945471075942 ,
 12.034180641174316, 21.690446261019574 ,
 5.782087802886963, 20.62817488125846 ,
 15.769731521606445, 90.69571574510753 ,
 33.718963623046875, 26.919675161933263 ,
 83.62213134765625, 127.70240111427617 ,
 6.789404392242432, 21.866396151294044 ,
 5.753351211547852, 19.199516657898602 ,
 6.913664817810059, 20.52009184251554 ,
 5.75408411026001 ,19.201457692829358 ,
 6.91972541809082 ,20.525744414861403 ,
 67.05557250976562, 159.4020687259188 ,
 42.75010681152344, 219.00227956490275 ,
 50.114349365234375, 100.09604570050267 ,
 18.588102340698242, 195.16034815407556 ,
 61.12921905517578, 123.71133508776707 ,
 4.853381633758545, 171.9024576125356 ,
 8.898716926574707, 115.49967272490017 ,
 2.032376289367676, 215.2122339521402 ,
 2.0561466217041016, 229.85472658165153 ,
 1.37593674659729 ,173.53463862892227 ,
 5.816436290740967, 43.990971363214086 ,
 46.36724853515625, 196.270219091633 ,
 14.018460273742676, 91.84052349962371 ,
 33.32183837890625, 102.77050537804053 ,
 39.36045837402344, 134.40530982188767 ,
 7.5668182373046875, 25.52133206287173 ,
 5.763470649719238, 42.87050528223145 ,
 7.579031467437744, 25.500584089309662 ,
 5.762180328369141, 42.84295849133453 ,
 7.581833839416504, 25.507802181105475 ,
 35.66175842285156, 265.07472129004685 ,
 37.60102081298828, 18.80381272476317 ,
 81.85986328125, 27.635317191812142 ,
 53.38380432128906, 217.46565479865086 ,
 65.6926498413086 ,41.83204918717321 ,
 75.23292541503906, 66.52549378433793 ,
 35.71003341674805, 264.66418857180486 ,
 35.698265075683594, 265.492752354852 ,
 35.69816207885742, 264.92651866489257 ,
 35.685028076171875, 265.05592293352265 ,
31.202465057373047 ,21.367574223463606 ,
 93.80559539794922 ,147.81505753024356 ,
 5.812340259552002, 17.5236186307364 ,
 47.83149719238281, 32.17994269445626 ,
 8.224610328674316 ,19.346631514598805 ,
 107.65904998779297, 68.09113129049123 ,
 5.803818702697754 ,17.568525838485744 ,
 102.32313537597656, 66.38145001335165 ,
 5.771985054016113 ,17.569837113461638 ,
 102.08616638183594, 65.84864251714978 ,
 67.18024444580078 ,206.58264010391747 ,
 6.326288223266602, 78.3401046795441 ,
 35.742523193359375 ,247.27752644129293 ,
 7.04774284362793, 76.6573802759249 ,
 18.189090728759766, 75.72712491102628 ,
 12.736583709716797, 92.670417687775 ,
 3.766432046890259, 42.96768977730716 ,
 23.302165985107422 ,469.98274410090136 ,
 52.558773040771484 ,110.06466147786332 ,
 3.889523983001709, 45.22809008928385 ,
 125.90839385986328, 28.08886948558152 ,
 68.9047622680664 ,25.574853903946547 ,
 24.972427368164062, 63.07347478084372 ,
 51.87223815917969, 44.90897275580961 ,
 29.80963897705078 ,23.580903173046366 ,
 85.22405242919922 ,213.67107646598618 ,
 91.70669555664062 ,139.80599069443136 ,
 55.11285400390625, 81.43226954866424 ,
 54.44743347167969, 93.69196444456644 ,
 68.25456237792969 ,121.22348572140586 ,
 47.256309509277344 ,36.206985301943924 ,
 79.6931381225586, 116.7509719298408 ,
 84.12281799316406, 70.7034439253379 ,
 18.133399963378906 ,35.549695596309085 ,
 112.05709075927734, 24.425196002782 ,
 63.8498649597168, 541.2837584521246 ,
 26.647335052490234, 29.79617722004043 ,
 20.02546501159668 ,29.718318976619052 ,
 23.44424057006836, 36.97439481945298 ,
 28.30899429321289 ,30.796530117704016 ,
 72.83624267578125, 300.2794135446859 ,
 64.81621551513672 ,29.428160428638158 ,
 21.00339126586914, 20.7766208748525 ,
 110.12342834472656, 55.915516169775 ,
 32.04602813720703, 362.8956179019327 ,
 46.15283966064453 ,30.943701640024884 ,
 21.034507751464844, 20.78732871839844 ,
 21.026777267456055, 20.64457994552376 ,
 21.02683448791504 ,20.644492090725407 ,
 21.02683448791504 ,20.644492795966006 ,
 37.68854522705078, 339.1126124730422 ,
 27.0136661529541, 158.2810818577163 ,
 4.887199878692627, 3544.051825072123 ,
 1.2420737743377686 ,153.45654868458786 ,
 64.14604187011719 ,240.91207938040634 ,
 39.441123962402344 ,254.43501428763346 ,
 8.853482246398926 ,156.03072004121623 ,
 1.1963226795196533 ,134.65339451088698 ,
 4.092845439910889, 214.4516091047761 ,
 1.2118875980377197 ,133.66683191623025 ,
 10.620016098022461 ,13.519174483108639 ,
 61.158546447753906 ,28.373066727985453 ,
 101.08427429199219, 27.8363225983914 ,
 133.6842041015625 ,26.208227493785035 ,
 10.436585426330566 ,17.377175818756108 ,
 126.79428100585938 ,28.487063225853284 ,
 10.421341896057129 ,17.374200064511353 ,
 127.02964782714844 ,28.487863519604314 ,
 10.42141342163086 ,17.374206121202114 ,
 127.0291976928711 ,28.487868878723198 ,
 19.80217933654785 ,2520.7611645690204 ,
 3.0849673748016357 ,233.66150141415835 ,
 2.971270799636841 ,17.715149607384145 ,
 2.1495983600616455, 4480.650225951484 ,
 19.018970489501953, 26.00225142087581 ,
 0.21451842784881592, 4609.038890249479 ,
 0.11489181965589523, 4613.118276480267 ,
 0.09876936674118042, 4619.031297786085 ,
 0.08668140321969986, 4621.514881914498 ,
 0.09865791350603104, 4617.831444066414 ,
 39.616092681884766 ,239.70549979491634 ,
 101.12605285644531 ,194.69869190947648 ,
 79.9365463256836, 307.8591456671927 ,
 3.452979564666748, 26.15060956085774 ,
 0.4429547190666199, 4486.576425682734 ,
 74.59956359863281, 279.3696248793686 ,
 0.7854398488998413, 4405.969347298148 ,
 0.7689741849899292 ,4408.8301180676435 ,
 0.7604063749313354, 4410.519894539122 ,
 0.8288967609405518 ,4394.5563143576155 ,
 7.840290069580078, 73.93524282052914 ,
 4.801275253295898 ,22.002919275738655 ,
 59.446990966796875 ,1278.5829158144456 ,
 42.03658676147461 ,140.37086686801464 ,
 32.57182693481445, 62.90612978174744 ,
 4.663174152374268 ,21.065791373189764 ,
 4.717075347900391, 20.40521405584915 ,
 4.688018798828125 ,20.398613529588268 ,
 4.670629978179932, 20.39787296789179 ,
 4.661810398101807, 20.39748134439433 ,]


array = [
 52.420921325683594 , 73.78874113466533 ,
 25.221759796142578 , 23.728925303723095 ,
 86.55220794677734 , 961.481649655873 ,
 41.67774200439453 , 149.52191386687545 ,
 71.35663604736328 , 98.14758095857661 ,
 54.80427551269531 , 139.11685265198463 ,
 16.02279281616211 , 622.0154194305409 ,
 46.922607421875 , 283.5587768548628 ,
 25.939821243286133 , 371.4327009287945 ,
 128.2727508544922 , 114.43885346406623 ,
 49.536155700683594 , 47.66714772023403 ,
 119.59403991699219 , 31.587183835727593 ,
 82.0447998046875 , 318.9510527840947 ,
 33.596717834472656 , 130.53800134482788 ,
 60.227294921875 , 86.40632153483176 ,
 136.6107177734375 , 67.93702187963842 ,
 31.704355239868164 , 24.446320363201664 ,
 34.272823333740234 , 213.55090585178542 ,
 46.26387023925781 , 52.78359687442168 ,
 25.492916107177734 , 86.41012866642708 ,
 19.81376838684082 , 26.025280314657365 ,
 17.412086486816406 , 2653.3988634920725 ,
 45.05118942260742 , 31.08400496689199 ,
 65.11859130859375 , 117.36789102463774 ,
 13.556816101074219 , 160.52629817102593 ,
 52.927825927734375 , 310.70689590709117 ,
 14.718057632446289 , 52.49684772810805 ,
 112.3964614868164 , 163.86596366480137 ,
 5.579408645629883 , 55.3498605772682 ,
 59.878482818603516 , 4031.979657316263 ,
 32.42317199707031 , 34.226054311432335 ,
 8.797791481018066 , 30.13663977116065 ,
 58.55976104736328 , 164.32670992130727 ,
 20.8143310546875 , 108.99565609421278 ,
 58.18614959716797 , 66.10569780001961 ,
 5.139261245727539 , 25.77084166406188 ,
 34.65256881713867 , 76.00941077400806 ,
 48.681617736816406 , 732.5794550821083 ,
 0.11320531368255615 , 4563.442297763113 ,
 5.263219833374023 , 23.975381039634676 ,
 44.80220031738281 , 395.7401761023093 ,
 63.931358337402344 , 176.1609106438286 ,
 58.7844352722168 , 257.2066280307488 ,
 37.05698776245117 , 122.14448619413044 ,
 0.8183916807174683 , 3103.3577655228146 ,
 24.201210021972656 , 123.06523877593871 ,
 2.9680933952331543 , 3906.979790724773 ,
 21.12438201904297 , 134.9687708635466 ,
 36.243717193603516 , 130.78302313736572 ,
 72.39132690429688 , 46.9794389724594 ,
 13.861368179321289 , 1642.7950499335811 ,
 5.636817932128906 , 87.96163411986848 ,
 42.56340026855469 , 327.56584700191877 ,
 6.543558120727539 , 25.372724530822072 ,
 31.681671142578125 , 67.49198200380593 ,
 35.18631362915039 , 24.22793014585 ,
 61.554054260253906 , 64.44913886246596 ,
 7.98638391494751 , 25.175261799411185 ,
 12.239633560180664 , 69.20775008241216 ,
 5.565981864929199 , 26.217666981853167 ,
 64.64794921875 , 66.97066430724749 ,
 36.499794006347656 , 275.2700878954398 ,
 54.14164733886719 , 102.68905176183189 ,
 81.31595611572266 , 89.77823358313614 ,
 40.00426483154297 , 61.125044716647636 ,
 25.209083557128906 , 27.169095208104928 ,
 31.439979553222656 , 61.76473225502956 ,
 2.9203009605407715 , 951.5894662385738 ,
 3.341170310974121 , 90.8595997696788 ,
 40.8555908203125 , 256.878020155626 ,
 0.192204087972641 , 4665.385460009993 ,
 39.522708892822266 , 79.2002895996186 ,
 4.731130123138428 , 3680.47212247129 ,
 14.942316055297852 , 25.391627871333117 ,
 8.05905532836914 , 24.50439274902027 ,
 19.651546478271484 , 240.43810560278757 ,
 57.464012145996094 , 26.468863364850556 ,
 28.379533767700195 , 679.0704340643354 ,
 5.628830432891846 , 2337.5373206739223 ,
 73.79537963867188 , 162.08643544656343 ,
 57.69282150268555 , 517.68888956974 ,
 13.248384475708008 , 75.47276917411727 ,
 12.83355712890625 , 2901.4032305920223 ,
 72.48875427246094 , 301.3713532881716 ,
 9.837937355041504 , 2269.1380789907203 ,
 43.040916442871094 , 126.12354119501387 ,
 22.779705047607422 , 165.56740532497483 ,
 49.494083404541016 , 307.93959080809043 ,
 32.38368606567383 , 329.18434151809623 ,
 20.294960021972656 , 264.09957344248676 ,
 62.625160217285156 , 99.72014092850868 ,
 24.078441619873047 , 23.579925219508645 ,
 149.88832092285156 , 91.55062676427463 ,
 10.324195861816406 , 95.67985394137548 ,
 19.060062408447266 , 102.55015669581813 ,
 48.21761703491211 , 176.53784633966657 ,
 17.487804412841797 , 2645.640069746284 ,
 114.94305419921875 , 97.24189770041704 ,
 50.248878479003906 , 145.15644259266713 ,
 51.5560302734375 , 199.86894232510764 ,
 48.654659271240234 , 22.821108885311734 ,
 86.65536499023438 , 171.8583405570531 ,
 55.107784271240234 , 627.7169706177284 ,
 52.39368438720703 , 91.37707546709925 ,
 21.865930557250977 , 103.75119338765683 ,
 19.900108337402344 , 519.549092586271 ,
 13.16749095916748 , 2240.5485625555266 ,
 74.32579040527344 , 293.71028574192496 ,
 60.26737976074219 , 332.63654815943426 ,
 9.16470718383789 , 998.4637228727684 ,
 34.06317901611328 , 29.990416068370603 ,
 14.167600631713867 , 97.33123555035894 ,
 19.078405380249023 , 155.27728689261193 ,
 56.35540771484375 , 60.802321543483856 ,
 28.999908447265625 , 31.718757972883086 ,
 32.343101501464844 , 312.55652757320576 ,
 29.23972511291504 , 113.2904382033496 ,
 73.47604370117188 , 118.33183282527008 ,
 73.1015853881836 , 113.96638692787903 ,
 61.78797912597656 , 33.85976612975152 ,
 40.26561737060547 , 380.2317801220638 ,
 70.86143493652344 , 534.261591553382 ,
 2.571906566619873 , 26.868186215151187 ,
 93.04801940917969 , 31.061674141653345 ,
 41.54649353027344 , 283.2338659492958 ,
 94.31912231445312 , 317.2260736800082 ,
 5.201263427734375 , 2451.747876288081 ,
 55.75932312011719 , 1099.2669460552306 ,
 58.67742919921875 , 209.7266991736792 ,
 29.610633850097656 , 354.8759756273072 ,
 135.8255157470703 , 23.11092253123578 ,
 7.026446342468262 , 34.896569694694364 ,
 253.1261444091797 , 100.74529970806087 ,
 21.77252197265625 , 127.3686926866796 ,
 7.965147495269775 , 25.56551026075757 ,
 39.78511047363281 , 81.73696238910912 ,
 7.430613994598389 , 2719.8129348020334 ,
 59.45301055908203 , 433.03378158123587 ,
 0.27690088748931885 , 4619.923920739951 ,
 33.954734802246094 , 23.321626486883773 ,
 3.8028013706207275 , 3880.302371119277 ,
 14.585588455200195 , 192.1974176595165 ,
 14.376900672912598 , 268.2294904985596 ,
 17.97848129272461 , 329.667806673942 ,
 52.12002182006836 , 412.0375060158407 ,
 14.167800903320312 , 49.62206588789339 ,
 39.84089279174805 , 266.0232955738109 ,
 36.13992691040039 , 178.49597721763953 ,
 27.306636810302734 , 209.62186293054276 ,
 36.818477630615234 , 930.4603482818252 ,
 57.00124740600586 , 139.6223731637818 ,
 46.22850799560547 , 158.0755314033375 ,
 53.03395462036133 , 278.307160101328 ,
 57.18016052246094 , 221.27876912110224 ,
 15.857223510742188 , 506.2359794175097 ,
 2.306471109390259 , 21.459435585374475 ,
 35.527862548828125 , 277.9448308272315 ,
 22.432998657226562 , 54.76291382662458 ,
 11.735189437866211 , 38.992721824726864 ,
 61.44309997558594 , 94.4429247454697 ,
 21.808500289916992 , 24.329126981400197 ,
 18.404125213623047 , 403.0627413592521 ,
 36.87770462036133 , 231.7654197920005 ,
 74.47473907470703 , 215.60392690290294 ,
 103.03961944580078 , 276.73129117087564 ,
 8.40452766418457 , 3267.9711954771146 ,
 3.5440895557403564 , 307.15377859880925 ,
 28.87076187133789 , 132.03131969297053 ,
 25.92361831665039 , 162.31842125697545 ,
 17.454790115356445 , 913.8645725327459 ,
 1.0675609111785889 , 4193.7607217463765 ,
 1.0990359783172607 , 2252.672402661031 ,
 16.676082611083984 , 323.78754118484346 ,
 40.150978088378906 , 89.50584682683365 ,
 79.37528228759766 , 74.75232534415977 ,
 43.48365783691406 , 64.72997426786493 ,
 12.064979553222656 , 125.28009998395456 ,
 139.45974731445312 , 41.38559539329349 ,
 62.16242218017578 , 125.77176879605241 ,
 1.7733471393585205 , 3611.437153666702 ,
 67.83071899414062 , 66.47358969893298 ,
 8.202984809875488 , 2553.6952866299507 ,
 42.9163818359375 , 24.169008190075143 ,
 44.01348876953125 , 87.57876641830725 ,
 0.10141192376613617 , 4602.57518182535 ,
 16.32601547241211 , 2876.190923759193 ,
 10.669529914855957 , 3819.859512735129 ,
 23.26407814025879 , 2372.7661590328753 ,
 3.489565134048462 , 331.6389849245621 ,
 30.02873420715332 , 52.10857549628144 ,
 57.40094757080078 , 374.6838700592042 ,
 56.7619743347168 , 306.02042980166624 ,
 7.556079864501953 , 30.266979418263748 ,
 74.88616180419922 , 170.81838566575178 ,
 14.327245712280273 , 157.67324721096682 ,
 21.808500289916992 , 24.329126981400197 ,
 57.47468185424805 , 227.25400316317928 ,
 62.6226921081543 , 193.7692303601442 ,
 51.84442901611328 , 142.59423874296306 ,
 33.32225036621094 , 245.5949503975141 ,
]

error_array = np.array(array).reshape(-1,2)
plt.figure()
# print(error_array[0,:5],np.array(array).reshape(-1,2)[:5,0])
plt.scatter(error_array[:,1],error_array[:,0])
max_length = int(np.max(error_array[:,0]))
for i in range(500):
 print(return_plots[i],i)
 plt.plot(np.ones(max_length)*(10*i+5),np.arange(max_length),alpha=return_plots[i]*10,linewidth=5,color='r')
plt.title('Prediction Error Mean')
plt.xlabel('Traj Return')
plt.ylabel('Uncertainty')
# plt.show()

std_array = [
 2.147037982940674, 44.74872457654477 ,
 1.8145502805709839 ,110.47292472304824 ,
 1.349924921989441, 17.82200475674651 ,
 1.629040002822876, 192.9055891206229 ,
 2.400341033935547 ,21.753699153105096 ,
 1.8027817010879517 ,192.18599181805865 ,
 1.3558361530303955 ,17.804225815000056 ,
 1.3479081392288208, 17.72422697843379 ,
 1.8164012432098389 ,184.71357631919605 ,
 1.3492577075958252 ,17.806580612459456 ,
 2.350792407989502, 420.2018704139108 ,
 1.215895652770996, 290.8056715672756 ,
 1.6791633367538452, 109.7312424536586 ,
 3.276362657546997, 721.0719989615964 ,
 3.12847900390625, 55.79410211125211 ,
 0.9662094116210938 ,24.865625089706356 ,
 0.8820129036903381 ,21.283520857352656 ,
 0.9046874642372131 ,21.139687806804673 ,
 0.8941341042518616, 20.93909281250643 ,
 0.888866662979126 ,21.058296297386896 ,
 1.7229650020599365 ,212.55734767446626 ,
 1.3711649179458618, 23.89690355636428 ,
 1.754385232925415, 62.88491784968949 ,
 1.9530692100524902, 30.27036149657124 ,
 1.7529526948928833 ,50.497721952519726 ,
 2.835320234298706, 49.38264631581291 ,
 1.3669407367706299, 23.78911000467083 ,
 2.987076997756958 ,50.309381932675876 ,
 1.3670786619186401 ,23.796313457458112 ,
 2.9683470726013184 ,50.165014185577085 ,
 1.8130489587783813, 89.49219969154218 ,
 1.286704421043396 ,20.362277811971264 ,
 1.197408676147461, 22.48793734911244 ,
 1.7862082719802856, 387.4824706527837 ,
 1.4820191860198975, 176.2372105065142 ,
 1.3470115661621094, 73.01614447697168 ,
 1.5366564989089966 ,318.99029976803905 ,
 1.6017982959747314 ,1171.8657831025198 ,
 1.661466121673584 ,265.06421349961744 ,
 2.010547161102295 ,136.62105068969626 ,
 1.1425639390945435 ,15.933141207103073 ,
 2.5324900150299072, 50.50108738610267 ,
 1.9367176294326782 ,27.040571958867957 ,
 1.8926814794540405 ,29.261003157590448 ,
 2.3976516723632812 ,103.92790371346834 ,
 1.7322965860366821, 30.15501611342001 ,
 1.1448540687561035 ,16.030973410139065 ,
 1.142249345779419 ,15.967036430187182 ,
 1.7324525117874146, 30.04850007684977 ,
 1.1448568105697632 ,16.031070135344443 ,
 2.376204252243042 ,4195.5662458526185 ,
 1.105017900466919 ,117.01230275485653 ,
 1.3054254055023193, 165.0966146858504 ,
 1.1788673400878906 ,169.37569444653326 ,
 0.9521000385284424 ,25.398964500994996 ,
 1.078019142150879, 68.15233797086293 ,
 1.0390868186950684, 28.52745366549243 ,
 0.9436659812927246 ,24.182963303919784 ,
 1.6841927766799927 ,26.811563684153967 ,
 1.0676430463790894 ,23.388820510015734 ,
 3.2133021354675293 ,14.486445869253128 ,
 1.6971534490585327 ,29.721469522614726 ,
 1.5536174774169922 ,30.297085051649457 ,
 1.1960163116455078, 16.58020215655312 ,
 1.8348983526229858, 28.82591770533004 ,
 1.734761118888855 ,28.460586986460257 ,
 1.1956638097763062, 16.57858200093576 ,
 1.7201063632965088 ,28.211984132393006 ,
 1.1956678628921509, 16.57858814681959 ,
 1.7200993299484253 ,28.211943056401253 ,
 2.362420082092285, 713.5065160975706 ,
 0.9716426730155945, 23.31508307533613 ,
 4.68204927444458, 421.259684590797 ,
 1.2020951509475708 ,27.244792095454006 ,
 1.7263562679290771, 87.10871286461723 ,
 0.8020492196083069 ,19.406247351809107 ,
 0.8234360218048096 ,19.951649644702684 ,
 0.8247842192649841, 19.96725657057203 ,
 0.8256043195724487 ,19.978650767765863 ,
 0.8256045579910278, 19.97138792654261 ,
 1.7499619722366333, 34.61410377729665 ,
 0.8807269930839539, 20.26892573921255 ,
 1.6260769367218018, 218.0248235979185 ,
 2.9975662231445312 ,117.70615488504994 ,
 2.674818277359009, 2881.630445144795 ,
 0.847347617149353, 19.75578970834776 ,
 0.851418673992157 ,19.804452961114173 ,
 0.8509715795516968, 19.80152422233399 ,
 0.8508529663085938, 19.80069702414137 ,
 0.8507931232452393, 19.79929247471479 ,
 1.6491923332214355 ,238.77277122246082 ,
 3.1287670135498047, 98.20928267873029 ,
 1.170045018196106, 354.2943789858281 ,
 2.585463047027588, 4583.234715880364 ,
 2.016824960708618 ,249.93309961379967 ,
 1.0566574335098267 ,131.32251962121666 ,
 0.7233303785324097, 249.367844400732 ,
 0.8449490666389465 ,23.043052131629988 ,
 0.9842045307159424 ,104.48765714379424 ,
 0.7946861386299133, 22.16648187051368 ,
 2.1497106552124023, 4157.328829833644 ,
 3.529188871383667, 26.561879019679942 ,
 1.8301457166671753, 29.482536118822857 ,
 1.3411136865615845, 172.38537255083423 ,
 1.3370723724365234, 138.95569547817274 ,
 2.635988712310791, 3364.6430805953933 ,
 3.212779998779297, 20.191508218035235 ,
 2.046567440032959, 2821.2802990148034 ,
 0.027357667684555054, 12.869099206505952 ,
 2.51352596282959, 661.9872133091524 ,
 3.0902199745178223, 189.83909408660315 ,
 0.037629012018442154, 12.787460467585834 ,
 0.02367422729730606, 12.710673299415657 ,
 2.51931095123291, 183.91365359072438 ,
 0.03922372683882713, 12.803043371656663 ,
 2.2275803089141846, 524.4733139288884 ,
 2.6595633029937744, 262.72586568640656 ,
 2.4924702644348145, 98.00120148418455 ,
 1.248187780380249, 132.46158609520987 ,
 1.1111689805984497, 105.64347204490403 ,
 1.113785982131958, 92.77040689095801 ,
 1.2510162591934204, 194.03905754632873 ,
 1.1490442752838135, 120.07193537936242 ,
 0.029473159462213516, 12.919057617249806 ,
 1.2266374826431274, 67.01148427113021 ,
 1.9790875911712646, 135.19236366464358 ,
 3.1844708919525146, 83.48897437762372 ,
 3.7623789310455322, 57.34416066392558 ,
 3.3292036056518555, 20.240796297889688 ,
 0.03343942016363144, 13.017001564790249 ,
 0.029062295332551003, 12.919934917616223 ,
 0.02786339819431305, 13.805296458438017 ,
 0.898406982421875, 28.340056331315395 ,
 1.37869393825531, 150.8757460207621 ,
 2.8383805751800537, 216.5072814514251 ,
 0.9158616662025452, 24.0503598666919 ,
 1.3914446830749512, 76.34380128003775 ,
 0.8100422620773315, 18.684351593277967 ,
 1.6912853717803955, 2757.227563509795 ,
 0.8127502799034119, 18.81847342817358 ,
 0.8149529695510864, 18.377073204976256 ,
 0.8139076828956604, 18.734675228073684 ,
 0.8149160146713257, 18.381578951421957 ,
 1.7004280090332031, 124.44124661215857 ,
 2.1924405097961426, 306.4504501710556 ,
 1.6834393739700317, 702.1167086446092 ,
 1.711116909980774, 22.547496120839387 ,
 2.5493316650390625, 654.8547271576178 ,
 2.0444095134735107, 161.30028570366966 ,
 1.6827441453933716, 622.1283570749117 ,
 2.6455276012420654, 1590.2088722343485 ,
 1.6522763967514038, 711.1366010454644 ,
 2.1284689903259277, 158.24130889170937 ,
 2.075839042663574, 20.838563100526617 ,
 1.9941380023956299, 208.55717742930267 ,
 3.789660930633545, 13.989424083687686 ,
 0.07295355200767517, 8.46838092428322 ,
 1.0016486644744873, 17.299238160486386 ,
 1.425411343574524, 29.958022282642254 ,
 1.9589730501174927, 26.485360737525895 ,
 1.8179333209991455, 28.490695829671182 ,
 2.030442953109741, 30.82304567983283 ,
 1.0034589767456055, 17.197386900447434 ,
 1.0023990869522095, 17.233091112343676 ,
 1.0023971796035767, 17.232698328432992 ,
 1.0024107694625854, 17.23279028378874 ,
 2.4929959774017334, 58.000079213175084 ,
 1.0401413440704346, 115.77779934340306 ,
 2.1909966468811035, 230.94676373369472 ,
 2.784916877746582, 22.033935160978974 ,
 2.46733021736145, 313.41884755153876 ,
 1.6833699941635132, 594.1344857401298 ,
 1.40035080909729, 64.17047754388943 ,
 3.5283782482147217, 71.41003912636808 ,
 3.186403751373291, 88.76649448037284 ,
 1.4258344173431396, 264.63915810387886 ,
]

std_array = [
 4.816032409667969 , 82.0581938160094 ,
 2.286972761154175 , 23.606362918006678 ,
 2.7063798904418945 , 27.931239703071775 ,
 0.9538664221763611 , 27.32375520203571 ,
 2.4789819717407227 , 159.60433655923669 ,
 1.918041467666626 , 371.206802224413 ,
 1.5489765405654907 , 52.322742369861984 ,
 5.709908962249756 , 111.2636893122442 ,
 1.6111925840377808 , 286.9828303290137 ,
 3.1792056560516357 , 147.9598019548838 ,
 2.643747329711914 , 4274.337471137297 ,
 2.2121798992156982 , 26.825089057414242 ,
 2.4367833137512207 , 2890.4437430419734 ,
 2.126110076904297 , 32.84878153036226 ,
 2.6484627723693848 , 258.72288491622055 ,
 1.927017331123352 , 151.31708125351835 ,
 1.7798101902008057 , 24.527124303190426 ,
 2.6460156440734863 , 66.87375889784926 ,
 2.1595542430877686 , 90.58111250465981 ,
 3.2518832683563232 , 169.72019067564796 ,
 2.7448325157165527 , 72.52671575757299 ,
 1.6845502853393555 , 72.07869956285577 ,
 2.033433437347412 , 284.1214596849834 ,
 3.767336845397949 , 74.44689561896332 ,
 3.0260417461395264 , 36.64739525028058 ,
 4.533794403076172 , 103.07450766368403 ,
 5.119377136230469 , 85.54424253220613 ,
 2.7923669815063477 , 4069.1110716203743 ,
 2.5082924365997314 , 4436.217765503869 ,
 2.538646936416626 , 312.0450379901656 ,
 5.11223030090332 , 50.115144283229164 ,
 3.4915947914123535 , 40.36342570855247 ,
 3.047196865081787 , 57.96083412565439 ,
 1.1444849967956543 , 122.65593444780026 ,
 1.9069055318832397 , 33.36092396699749 ,
 2.423927068710327 , 4585.801575075515 ,
 5.691041946411133 , 108.19231105712578 ,
 1.462558388710022 , 26.44146715164959 ,
 1.7232967615127563 , 68.11000226676876 ,
 2.619767665863037 , 320.9954886174227 ,
 3.6202445030212402 , 62.80270935598017 ,
 3.5198187828063965 , 428.6609102327809 ,
 2.5550642013549805 , 3485.658610748304 ,
 2.6599984169006348 , 4490.234020895601 ,
 3.612563133239746 , 161.27790857635665 ,
 2.100593328475952 , 1084.019370135855 ,
 1.2694398164749146 , 44.36431667302804 ,
 2.6638147830963135 , 25.23896110760519 ,
 3.345214366912842 , 118.59169654709714 ,
 2.540851354598999 , 4027.224867891877 ,
 1.2489464282989502 , 89.88324571253548 ,
 3.562364101409912 , 173.7103154235027 ,
 3.6891212463378906 , 25.346826188610354 ,
 1.4631550312042236 , 24.380660365288932 ,
 1.3449585437774658 , 3224.7955954893214 ,
 1.1317750215530396 , 64.67674465721666 ,
 2.2850303649902344 , 958.7852082973754 ,
 1.5386227369308472 , 23.246115730331304 ,
 3.747403144836426 , 78.99089791980597 ,
 1.6928551197052002 , 25.71975981740474 ,
 6.50954008102417 , 66.97066430724749 ,
 2.093782901763916 , 379.94633068631146 ,
 3.5431759357452393 , 135.86812117307844 ,
 4.103983402252197 , 72.25846025632711 ,
 4.2116241455078125 , 3799.593304415055 ,
 2.707810163497925 , 4256.70688577319 ,
 2.6998343467712402 , 80.18290518664081 ,
 2.666037082672119 , 1925.5756099419064 ,
 3.8205599784851074 , 96.95295512579156 ,
 1.2701876163482666 , 25.537110358711903 ,
 2.0649445056915283 , 409.4343187378006 ,
 2.3051531314849854 , 4562.714670144183 ,
 3.6410348415374756 , 79.26498257123076 ,
 1.1647710800170898 , 124.18004487410481 ,
 1.9399114847183228 , 210.50697535247298 ,
 2.7230801582336426 , 156.32877739684386 ,
 3.3785178661346436 , 198.81626748012366 ,
 2.720560073852539 , 163.02918420970104 ,
 2.1939072608947754 , 438.9228153078502 ,
 2.5907514095306396 , 106.326172724732 ,
 2.3780550956726074 , 517.68888956974 ,
 3.034275770187378 , 305.15699818101206 ,
 2.5667097568511963 , 68.72159285020737 ,
 2.1356794834136963 , 241.91735974689394 ,
 3.715883255004883 , 132.68682137107686 ,
 2.6801671981811523 , 26.125078526512986 ,
 4.540114879608154 , 58.90748655684898 ,
 2.150737762451172 , 242.3726631246147 ,
 3.9838757514953613 , 224.78144286451464 ,
 2.644120216369629 , 4525.369308679528 ,
 3.114604949951172 , 62.31970517100298 ,
 1.824139952659607 , 23.123517868278288 ,
 3.251448631286621 , 311.0924363707836 ,
 2.8732235431671143 , 209.73360309567897 ,
 2.112891674041748 , 89.01584416467739 ,
 4.514644145965576 , 82.55587603807938 ,
 2.379141092300415 , 2644.353188526784 ,
 6.667595386505127 , 37.04476439310797 ,
 2.775114059448242 , 61.07829269088762 ,
 3.029207468032837 , 79.4114265006823 ,
 2.410857915878296 , 1894.738576393346 ,
 5.728794097900391 , 1583.0475285078342 ,
 2.54592227935791 , 24.31575910602818 ,
 0.9777740240097046 , 30.518246659097812 ,
 2.8117339611053467 , 81.31473744827338 ,
 1.2271963357925415 , 376.7850249625475 ,
 3.3660924434661865 , 578.6220339631109 ,
 1.921047329902649 , 298.14024877019466 ,
 2.162609100341797 , 4649.219350821797 ,
 4.051411151885986 , 362.9545185854937 ,
 1.79533851146698 , 27.08948698995645 ,
 2.0036492347717285 , 4548.592753256421 ,
 0.7986542582511902 , 31.172859446107626 ,
 2.64866304397583 , 259.61896978823506 ,
 3.8175766468048096 , 71.91564333300235 ,
 1.5462746620178223 , 291.0915796653458 ,
 2.3029870986938477 , 96.31274693085646 ,
 2.381669282913208 , 504.7007363966357 ,
 1.6943588256835938 , 28.52073478775042 ,
 1.2650744915008545 , 277.8786700949979 ,
 1.0425496101379395 , 26.865403169942162 ,
 6.093600749969482 , 118.39262029266371 ,
 2.994276285171509 , 292.91417553979363 ,
 2.1975741386413574 , 27.066204103780894 ,
 3.413092613220215 , 247.35188197069013 ,
 4.292870044708252 , 408.59642128917494 ,
 2.2416505813598633 , 2018.5194956228738 ,
 1.4819226264953613 , 215.3662420791361 ,
 2.493398427963257 , 126.9224271751596 ,
 5.392279624938965 , 56.93198536553204 ,
 2.0886032581329346 , 21.803074980650603 ,
 1.664992332458496 , 31.101752913784438 ,
 1.8356856107711792 , 21.969929044244523 ,
 2.5056753158569336 , 4016.232390134078 ,
 1.382171630859375 , 29.027928815937663 ,
 1.165432095527649 , 340.525622570353 ,
 2.3135898113250732 , 4622.834921216534 ,
 2.1664700508117676 , 135.89797129700167 ,
 3.5611636638641357 , 47.84739566657631 ,
 1.663723111152649 , 454.5885640148771 ,
 5.772871971130371 , 32.93297001552866 ,
 3.049931526184082 , 4459.882252955562 ,
 1.1658607721328735 , 211.72493135632476 ,
 1.3346562385559082 , 188.474078203854 ,
 2.226925849914551 , 3322.98265124225 ,
 3.1963462829589844 , 179.24069273862443 ,
 0.9730031490325928 , 192.5454180166114 ,
 3.9344937801361084 , 155.80049118458726 ,
 2.631863594055176 , 3322.248379872679 ,
 2.590348482131958 , 256.5881351463089 ,
 3.552241802215576 , 72.75255777592082 ,
 2.4066596031188965 , 349.1114913783333 ,
 2.304919719696045 , 4495.358139737113 ,
 3.911802053451538 , 67.29745374143369 ,
 5.142807483673096 , 76.93279200740857 ,
 4.521914482116699 , 88.30162280047774 ,
 0.8631387948989868 , 31.00082241294975 ,
 2.2159807682037354 , 207.99378756622514 ,
 2.4589250087738037 , 29.334857113627578 ,
 2.4746623039245605 , 173.51178086292137 ,
 2.268131732940674 , 413.85842326251463 ,
 1.459969401359558 , 23.15205359711801 ,
 4.499299049377441 , 146.0206633133485 ,
 1.650895118713379 , 125.62649230213441 ,
 1.3273507356643677 , 381.7667601302585 ,
 2.27323055267334 , 1816.6035848328982 ,
 2.1996288299560547 , 4366.569153990229 ,
 2.1664931774139404 , 3006.893393590055 ,
 2.5216000080108643 , 24.32300409707283 ,
 3.8101887702941895 , 71.93992700005919 ,
 2.677917718887329 , 62.629535696848265 ,
 1.7225652933120728 , 34.36056103164742 ,
 3.0160422325134277 , 332.00752382801534 ,
 1.371058464050293 , 155.1375830171899 ,
 3.4086527824401855 , 173.45013646226033 ,
 3.0369346141815186 , 74.65108879199214 ,
 1.5430647134780884 , 90.70632354397978 ,
 3.6548044681549072 , 549.292841869456 ,
 1.529491662979126 , 173.96246039442798 ,
 1.9773497581481934 , 99.69555645781887 ,
 2.307990312576294 , 32.09732674005514 ,
 1.7199214696884155 , 101.25978012815946 ,
 0.9482594728469849 , 66.93895848991954 ,
 1.3229749202728271 , 224.73158145368785 ,
 2.2180263996124268 , 24.181069801821724 ,
 2.2114946842193604 , 54.1953608369723 ,
 1.2730169296264648 , 304.0849009472746 ,
 2.095130681991577 , 896.1145745211734 ,
 3.509873151779175 , 58.80809369126507 ,
 2.1576013565063477 , 3957.026467384987 ,
 1.5949980020523071 , 295.097281594385 ,
 2.3622870445251465 , 1123.7549450525744 ,
 4.712616920471191 , 220.56517418461385 ,
 0.8520463705062866 , 386.583645775194 ,
 3.3647801876068115 , 398.04179225986263 ,
 1.3530081510543823 , 318.147677037417 ,
 1.4569183588027954 , 23.140519649337712 ,
 2.3702478408813477 , 94.38349297601987 ,
 3.1537575721740723 , 231.58805308744775 ,
 0.8463342785835266 , 36.15577491297792 ,
]

std_array = np.array(std_array).reshape(-1,2)
plt.figure()
plt.scatter(std_array[:,1],std_array[:,0])
max_length = int(np.max(std_array[:,0]))+1
for i in range(500):
 print(return_plots[i],i)
 plt.plot(np.ones(max_length)*(10*i+5),np.arange(max_length)/4*4.5,alpha=return_plots[i]*10,linewidth=5,color='r')
plt.title('Prediction Std')
plt.xlabel('Traj Return')
plt.ylabel('Uncertainty')
plt.show()