import csv
import sys
import ssl
import urllib.request
from bs4 import BeautifulSoup

def main():

    htmlUrl = "https://www.enchantedlearning.com/wordlist/"
    # Bypass SSL context of the shtml
    req = urllib.request.Request(htmlUrl, headers={ 'X-Mashape-Key': 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' })
    gcontext = ssl.SSLContext()

    page = urllib.request.urlopen(req, context=gcontext).read()
    soup = BeautifulSoup(page)

    test = soup.select("a[href*=wordlist][href*=shtml]")

    catList = str(test).split('</a>')
    for y in range(0, len(catList)):
        # removes anchor references
        catList[y] = catList[y].replace(', <a href="', '')
        catList[y] = catList[y].split('" target', 1)[0]

    # removed links from top of page for enchanted learning site
    del catList[0:6]
    del catList[len(catList)-1]

    # print(catList)
    getCategoryList(catList)

    

def getCategoryList(categoryList):

    for i in range(0, len(categoryList)):
        htmlUrl = "https://www.enchantedlearning.com" + categoryList[i]
        # Bypass SSL context of the shtml
        req = urllib.request.Request(htmlUrl, headers={ 'X-Mashape-Key': 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' })
        gcontext = ssl.SSLContext()

        page = urllib.request.urlopen(req, context=gcontext).read()
        soup = BeautifulSoup(page)

        test = soup.find_all(class_="wordlist-item")

        itemList = str(test).split(',')
        for y in range(0, len(itemList)):
            # removes div references
            itemList[y] = itemList[y].replace(' <div class="wordlist-item">', '')
            itemList[y] = itemList[y].replace('[<div class="wordlist-item">', '')
            itemList[y] = itemList[y].replace('</div>', '')
            itemList[y] = itemList[y].replace(']', '')
            # removes anchor references
            itemList[y] = itemList[y].replace('</a>', '')
            if '>' in itemList[y]:
                itemList[y] = itemList[y].split('>', 1)[1]

        fileName = categoryList[i].replace('/wordlist', './word_list_csv')
        fileName = fileName.replace('shtml', 'csv')
        
        # output list to csv file for feature comparison
        outputCsvFile = fileName
        with open( outputCsvFile, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(itemList)
        
        print('finished computing file -> ', fileName)
    
    print('finished computing csv files')


if __name__ == "__main__":
    main()
