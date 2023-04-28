songs = dir('*.mp3');
writecell({songs.name},"names.txt");
num_files = size(songs,1);
num_features = 10;
x=zeros(num_files);
for i=1:num_files
      for j=1:num_features
          x(i,j)=i+j;
      end
end
song_count = 1;
for song = songs'
    song_mir = miraudio(song.name);
    %x(song_count, 1) = string(song.name);
    x(song_count, 1) = mirgetdata(mircentroid(song_mir));
    x(song_count, 2) = mirgetdata(mirzerocross(song_mir));
    x(song_count, 3) = mirgetdata(mirrms(song_mir));
    x(song_count, 4) = mirgetdata(mirlowenergy(song_mir));
    x(song_count, 5) = mirgetdata(mirrolloff(song_mir));
    x(song_count, 6) = mirgetdata(mirbrightness(song_mir));
    x(song_count, 7) = mirgetdata(mirkurtosis(song_mir));
    x(song_count, 8) = mirgetdata(mirflatness(song_mir));
    x(song_count, 9) = mirgetdata(mirentropy(song_mir));
    x(song_count, 10) = mirgetdata(mirspread(song_mir));
    clear("song_mir");
    song_count = song_count + 1;
    % Do some stuff
end
dlmwrite("data.txt",x,'precision',20,'delimiter',' ');