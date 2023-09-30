# import pandas lib as pd
import pandas as pd
import numpy as np

cols_buscap = [0,2,8,9,10]
cols_loaddata = [0,6,7]
cols_gendata = [0]

# only read specific columns from an excel file
df = pd.read_excel('Linedata.xls', usecols=cols_buscap)
df2 = pd.read_excel('Loaddata.xls', usecols=cols_loaddata)
df3 = pd.read_excel('GenData.xls', usecols=cols_gendata)


# print(df2.dropna())


busparams = []
lineparams = []
ldparams=[]
i=2
# print((df['B'].where(df['From Number'] == i ).dropna()).tolist())

while i <= 118:
    # print((df['B'].where(df['From Number'] == i ).dropna()).tolist())
    bus_suscept1=sum((df['B'].where(df['From Number'] == i ).dropna()).tolist())
    bus_suscept2=sum((df['B'].where(df['To Number'] == i ).dropna()).tolist())
    bus_suscept=bus_suscept1 + bus_suscept2
    # print(bus_suscept)
    busparams.append(bus_suscept)
    i = i + 1

i = 0
while i < len(df.index):
    line_rx = df.values[i].tolist()
    lineparams.append(line_rx[2])
    lineparams.append(line_rx[3])
    i = i + 1


i = 0
while i < len(df2.index):
    bus_ld = df2.values[i].tolist()
    ldparams.append(bus_ld[1])
    ldparams.append(bus_ld[2])
    i = i + 1

busparams = np.array((busparams), dtype=float)
lineparams = np.array((lineparams), dtype=float)
ldparams = np.array((ldparams), dtype=float)


#
# print(busparams)
# print(lineparams)
# print(ldparams)

i = 0
while i < len(df.index):
    bus_suscept = df.values[i].tolist()
    print('solline%s_%s=Line.linemodel(line%s_%s,wconv01,pline%s_%s,bus%s,bus%s)'%(str(int(bus_suscept[0])).zfill(3),(str(int(bus_suscept[1])).zfill(3)),
                                                                                   str(int(bus_suscept[0])).zfill(3),(str(int(bus_suscept[1])).zfill(3)),
                                                                                   str(int(bus_suscept[0])).zfill(3),(str(int(bus_suscept[1])).zfill(3)),
                                                                                   str(int(bus_suscept[0])).zfill(3),(str(int(bus_suscept[1])).zfill(3))))
    i = i + 1

i=2
while i < len(df.index):

    bus_arriving=df['From Number'].where(df['To Number'] == i ).dropna().tolist()
    bus_leaving=df['To Number'].where(df['From Number'] == i ).dropna().tolist()
    bus_load=df2['Number of Bus'].where(df2['Number of Bus'] == i ).dropna().tolist()
    bus_gen=df3['Gen Number of Bus'].where(df3['Gen Number of Bus'] == i ).dropna().tolist()


    # print(bus_gen)
    # print(bus_arriving)
    # print(bus_leaving)

    print('solbus%s=Bus.busmodel(bus%s,wconv01,pbus%s[0],'%(str(i).zfill(3),str(i).zfill(3),str(i).zfill(3)),end="")


    if bus_load:
        print('pld%s,'% (str(int(bus_load[0])).zfill(3)),end="")
    else:
        print('null,',end="")


    if bus_gen:
        print('GenIQ%s[0],GenID%s[0],' % (str(int(bus_gen[0])).zfill(3), str(int(bus_gen[0])).zfill(3)),end="")
    else:
        print('null,null,',end="")

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
    i=i+1

i=0
while i < len(df2.index):
    bus_ldno = df2.values[i].tolist()
    print('solload%s=Indload.Lloadmodel(Lload%s,wconv38,pld%s[1],bus%s)'%(str(int(bus_ldno[0])).zfill(3),str(int(bus_ldno[0])).zfill(3),
          str(int(bus_ldno[0])).zfill(3),str(int(bus_ldno[0])).zfill(3)))
    i=i+1
# solLload03 = Indload.Lloadmodel(Lload03, wconv38, pld03[1], bus03)
i=0

while i < len(df3.index):
    gen_no = df3.values[i].tolist()
    print('solu%s=unit1.unitmodel(gen%s,pgen%s,inputs%s,bus%s[0],bus%s[1],delta%s[0])'%(str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3),
                                                                                        str(int(gen_no[0])).zfill(3)))
    i=i+1