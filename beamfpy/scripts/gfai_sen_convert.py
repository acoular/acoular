f=open('Ring32_MicPos.sen')
lines=f.readlines()
i=-3
for line in lines:
    tokens=line.split(':')
    if tokens[0]=='xPos':
        print '    <pos Name="Point %i" x="%s'%(i,tokens[1][:-2]),
    if tokens[0]=='yPos':
        print '" y="%s" z="0"/>' % tokens[1][:-2]
        i+=1
i=-2      
for line in lines:
    tokens=line.split(':')
    if tokens[0]=='Trans':
        print '	<pos Name="Point %i" factor="%s"/>' % (i,tokens[1][:-2])
        i+=1
f.close()
