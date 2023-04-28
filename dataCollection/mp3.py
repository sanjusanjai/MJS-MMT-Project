import csv
import re
import youtube_dl
import requests
from bs4 import BeautifulSoup
from operator import itemgetter
import ffmpeg
from pydub import AudioSegment
import subprocess
import pickle

# environment todo: remove these fucking useless packages and lock rpm fusion again
# from tube_dl import Youtube
# from mutagen.mp3 import MP3
# from mutagen.easyid3 import EasyID3

def download_ytvid_as_mp3(ytLink,i,j,path):    
    
    # try:
    video_info = youtube_dl.YoutubeDL().extract_info(url = ytLink,download=False)      

    videoName=re.findall(r'\w+',video_info['title'])[:2]
    videoName="_".join(videoName)
    filePath=path+str(i)+"_"+str(j)+"_"+videoName        
    options={
        'format': 'bestaudio/best',
        'outtmpl':filePath+".mp4",
        'keepvideo':False, 
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])                
    # convert file codec from mp4 formatting to actual mp3 codec instead of aac cs fucking youtube-dl wont fucking do it
    try:
        subprocess.run("ffmpeg -i {}.mp4 {}.mp3".format(filePath,filePath),shell=False)        
    except:
        pass
    # trim the mp3 so only mid 2m is present in output
    sound = AudioSegment.from_mp3(filePath+".mp3")
    trimmed= sound[int(len(sound)/2-60000):int(len(sound)/2+60000)]
    trimmed.export(filePath+".mp3",format="mp3")

    # except:
        # print("oops probably unloadable for some reason idk why need to look into it")        

# function to get sorted links
def sortedLinks(links,year):    
    temp=[]
    lmao=0
    for link in links:
        try:            
            print(link)
            soup = BeautifulSoup(requests.get(link).text, 'lxml')
            views=soup.select_one('meta[itemprop="interactionCount"][content]')['content']
            # views=int(subprocess.getoutput("./ytviews.sh "+link))
            # duration=youtube_dl.YoutubeDL().extract_info(url = link,download=False)['duration']      
            # if(duration<150):
            #     continue        
            temp.append({"link":link,"views":int(views)})
        except:
            print("oops couldn't get info about link"+" "+link)
            lmao+=1
    temp=sorted(temp,key=itemgetter('views'),reverse=True)
    for i in range(len(temp)):
        for j in range(i+1,len(temp)):
            if(temp[i]["link"]==temp[j]["link"]):
                print("aye yo fuck")
    linkPath="./csvData/"
    linkList=[]
    for link in temp:
        print(link)
        linkList.append(link["link"])
    with open(linkPath+"links_"+str(year),"wb") as fp:
        pickle.dump(linkList,fp)
    print("year {} {}".format(year,len(linkList)))
    return temp,lmao

relPath="./csvData/song_details_"
for i in range(2000,2022):    
    fileName=relPath+str(i)+".csv"     
    links=[]
    with open(fileName,'r') as f:
        temp=csv.reader(f)        
        for row in temp:                        
            links.append(row[-1])
    links=links[1:]

    # remove duplicates cs there a lot for some reason

    temp=[]
    for link in links:
        if link not in temp:
            temp.append(link)
    links=temp
    # print("year {} {}".format(i,len(links)))
    # get the sorted links
    links,lmao=sortedLinks(links,i)
# pprint(links)
# print(lmao)
# for j in range(1,31):
#     download_ytvid_as_mp3(links[j]["link"],i,j,"./mp3files/top/song_details_")
# for j in range(len(links)-1,len(links)-31,-1):
#     download_ytvid_as_mp3(links[j]["link"],i,len(links)-j,"./mp3files/bottom/song_details_")
