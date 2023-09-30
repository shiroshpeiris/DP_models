# import pandas lib as pd
import pandas as pd
import numpy as np

cols_buscap = [0,2]
cols_loaddata = [0,6,7]
cols_gendata = [0]

# only read specific columns from an excel file
df = pd.read_excel('Linedata.xls', usecols=cols_buscap)
df2 = pd.read_excel('Loaddata.xls', usecols=cols_loaddata)
df3 = pd.read_excel('GenData.xls', usecols=cols_gendata)




i = 0
bus_arr=[]
while i < len(df.index):
    bus_inout = df.values[i].tolist()
    bus_arr = (np.append(bus_arr,bus_inout[0:2]))
    i = i + 1
bus_arr = bus_arr.astype(int)

print(bus_arr)


i = 0
ldbus_no_arr=[]
while i < len(df2.index):
    bus_ldno = df2.values[i].tolist()
    ldbus_no_arr = (np.append(ldbus_no_arr,bus_ldno[0]))
    i = i + 1
ldbus_no_arr = ldbus_no_arr.astype(int)

print(ldbus_no_arr)

i=0
genbus_no_arr=[]
while i < len(df3.index):
    gen_no = df3.values[i].tolist()
    genbus_no_arr = (np.append(genbus_no_arr,gen_no[0]))
    i = i + 1
genbus_no_arr = genbus_no_arr.astype(int)

print(genbus_no_arr)


i=1
genbus_ind = []
loadbus_ind=[]

while i <= 118:
    bus_load=df2['Number of Bus'].where(df2['Number of Bus'] == i ).dropna().tolist()
    bus_gen=df3['Gen Number of Bus'].where(df3['Gen Number of Bus'] == i ).dropna().tolist()

    if bus_load:
        loadbus_ind.append(1)
    else:
        loadbus_ind.append(0)


    if bus_gen:
        genbus_ind.append(1)
    else:
        genbus_ind.append(0)
    i = i+1


# print(busparams)
# print(lineparams)
# print(ldparams)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##=====================Line functions generation=========================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


i=0
while i < len(df.index):
    bus_suscept = df.values[i].tolist()
    print('solline[%s:%s]=Line.linemodel(line%s_%s,wconv01,pline%s_%s,bus%s,bus%s)'%(str(i*2),(str(i*2+2)),
                                                                                   str(int(bus_suscept[0])).zfill(3),(str(int(bus_suscept[1])).zfill(3)),
                                                                                   str(int(bus_suscept[0])).zfill(3),(str(int(bus_suscept[1])).zfill(3)),
                                                                                   str(int(bus_suscept[0])).zfill(3),(str(int(bus_suscept[1])).zfill(3))))
    print('solline[%s:%s]=Line.linemodel(lines[%s:%s],wconv01,p_line[%s:%s],buses[%s:%s],buses[%s:%s])'%(str(i*2),(str(i*2+2)),
                                                                                   str(i*2),(str(i*2+2)),
                                                                                   str(i*2),(str(i*2+2)),
                                                                                   str(bus_arr[i*2]*2-2),(str(bus_arr[i*2]*2)),
                                                                                   str(bus_arr[i*2+1]*2-2),(str(bus_arr[i*2+1]*2))))


    # bus_arr = np.append(bus_arr,bus_suscept[0:2])
    i = i + 1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##=====================Bus functions generation=========================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i=1
j=0
l=0

while i < len(df.index):
    bus_arriving=df['From Number'].where(df['To Number'] == i ).dropna().tolist()
    bus_leaving=df['To Number'].where(df['From Number'] == i ).dropna().tolist()
    bus_load=df2['Number of Bus'].where(df2['Number of Bus'] == i ).dropna().tolist()
    bus_gen=df3['Gen Number of Bus'].where(df3['Gen Number of Bus'] == i ).dropna().tolist()


    # print(bus_gen)
    # print(bus_arriving)
    # print(bus_leaving)

    print('solbus[%s:%s]=Bus.busmodel(bus%s,wconv1,pbus%s[0],'%(str((i-1)*2),str((i-1)*2+2),str(i).zfill(3),str(i).zfill(3)),end="")


    if bus_load:
        print('pld%s,'% (str(int(bus_load[0])).zfill(3)),end="")
    else:
        print('null,',end="")


    if bus_gen:
        print('GenIQ%s[0],GenID%s[0],' % (str(int(bus_gen[0])).zfill(3), str(int(bus_gen[0])).zfill(3)),end="")
    else:
        print('0,0,',end="")

    if bus_load:
        print('Lload%s,'% (str(int(bus_load[0])).zfill(3)),end="")
    else:
        print('null,',end="")


    k=0
    if bus_arriving:
        while k < len(bus_arriving):
            print('line%s_%s'%(str(int(bus_arriving[k])).zfill(3),str(i).zfill(3)),end="")
            if k<len(bus_arriving) - 1:
                print('+',end="")
            k = k+1
    else:
        print("null",end="")

    print(",",end="")


    k=0
    if bus_leaving:
        while k < len(bus_leaving):
            print('line%s_%s'%(str(i).zfill(3),str(int(bus_leaving[k])).zfill(3)),end="")
            if k<len(bus_leaving) - 1:
                print('+',end="")
            k = k+1
    else:
        print("null",end="")

    print(")")



# -------------------------------alternative------------------------------------------
# ###############################################################################################



    print('solbus[%s:%s]=Bus.busmodel(buses[%s:%s],wconv[0],p_buses[%s],'%(str((i-1)*2),str((i-1)*2+2),str((i-1)*2),str((i-1)*2+2),str(i-1)),end="")


    if bus_load:
        print('p_loads[%s:%s],'% (str(int((j)*2)),str(int((j)*2+2))),end="")
    else:
        print('null,',end="")


    if bus_gen:
        print('GenIQ[%s],GenID[%s],' % (str((l)), str((l))),end="")
        l = l + 1
    else:
        print('0,0,',end="")

    if bus_load:
        print('loads[%s:%s],'% (str(int((j)*2)),str(int((j)*2+2))),end="")
        j = j + 1
    else:
        print('null,',end="")


    k=0
    if bus_arriving:
        while k < len(bus_arriving):
            print('line%s_%s'%(str(int(bus_arriving[k])).zfill(3),str(i).zfill(3)),end="")
            if k<len(bus_arriving) - 1:
                print('+',end="")
            k = k+1
    else:
        print("null",end="")

    print(",",end="")


    k=0
    if bus_leaving:
        while k < len(bus_leaving):
            print('line%s_%s'%(str(i).zfill(3),str(int(bus_leaving[k])).zfill(3)),end="")
            if k<len(bus_leaving) - 1:
                print('+',end="")
            k = k+1
    else:
        print("null",end="")

    print(")")


    i=i+1

print(genbus_ind[:118])
print(len(genbus_ind[:118]))
print(loadbus_ind[:118])
print(len(loadbus_ind[:118]))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##=====================Load functions generation=========================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i = 0
while i < len(df2.index):
    bus_ldno = df2.values[i].tolist()
    print('solload%s=Indload.Lloadmodel(Lload%s,wconv1,pld%s[1],bus%s)'%(str(int(bus_ldno[0])).zfill(3),str(int(bus_ldno[0])).zfill(3),
          str(int(bus_ldno[0])).zfill(3),str(int(bus_ldno[0])).zfill(3)))

    print('solload[%s:%s]=Indload.Lloadmodel(Lload[%s:%s],wconv[0],loads[%s],bus[%s:%s])'%(str(i*2),(str(i*2+2)),str(i*2),(str(i*2+2)),
                                                                                       str(i*2+1),
                                                                                       str(int(ldbus_no_arr[i]*2-2)),
                                                                                       str(int(ldbus_no_arr[i]*2))))



    i = i + 1



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##=====================Source functions generation=========================================
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i = 0
while i < len(df3.index):
    gen_no = df3.values[i].tolist()
    print('solu%s=unit1.unitmodel(gen%s,pgen%s,inputs%s,bus%s[0],bus%s[1],delta%s[0])'%(str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3)))

    print('gensols[%s:%s]=unit1.unitmodel(gen[%s:%s],pgen[%s:%s],inputs[%s:%s],buses[%s],buses[%s],angles[%s])'%(str(i*2),(str(i*2+2)),
                                                                                        str(i*2),(str(i*2+2)),
                                                                                        str(i*4),(str(i*4+4)),
                                                                                        str(i*4),(str(i*4+4)),
                                                                                        str(int(gen_no[0]*2-2)),
                                                                                        str(int(gen_no[0]*2-1)),
                                                                                        str(i-1)))

    i = i + 1