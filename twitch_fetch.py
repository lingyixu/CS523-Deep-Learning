import socket
import pandas as pd
from datetime import datetime
import time
import re
import logging
from emoji import demojize

def get_chat_dataframe(file):
    data = []

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n\n')
        
        for line in lines:
            try:
                time_logged = line.split('—')[0].strip()
                time_logged = datetime.strptime(time_logged, '%Y-%m-%d_%H:%M:%S')

                username_message = line.split('—')[1:]
                username_message = '—'.join(username_message).strip()

                username, channel, message = re.search(
                    ':(.*)\!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)', username_message
                ).groups()

                d = {
                    'dt': time_logged,
                    'channel': channel,
                    'username': username,
                    'message': message
                }

                data.append(d)
            
            except Exception:
                pass
            
    return pd.DataFrame().from_records(data)
server = 'irc.chat.twitch.tv'
port = 6667
nickname = 'ec523proj'
token = 'oauth:v6hzj3s7pqdk1n37c90zuee1pcgq69'
channel = '#loltyler1'

sock = socket.socket()
sock.connect((server, port))
sock.send(f"PASS {token}\n".encode('utf-8'))
sock.send(f"NICK {nickname}\n".encode('utf-8'))
sock.send(f"JOIN {channel}\n".encode('utf-8'))

#resp = sock.recv(2048).decode('utf-8')
#print(resp)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s — %(message)s',
                    datefmt='%Y-%m-%d_%H:%M:%S',
                    handlers=[logging.FileHandler('chat.log', encoding='utf-8')])
#logging.info(resp)

try:
    while True:
        socket.setdefaulttimeout(1)
        time.sleep(1)
        resp = sock.recv(2048).decode('utf-8')

        if resp.startswith('PING'):
            # sock.send("PONG :tmi.twitch.tv\n".encode('utf-8'))
            sock.send("PONG\n".encode('utf-8'))
        elif len(resp) > 0:
            logging.info(demojize(resp))
            df = get_chat_dataframe('chat.log')


except KeyboardInterrupt:
    sock.close()

df.set_index('dt', inplace=True)
print(df)
df.head()
df.to_csv((channel+'.csv'))

