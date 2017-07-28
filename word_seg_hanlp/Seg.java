
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.CRF.CRFSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
/**
 * 第一个Demo，惊鸿一瞥
 *
 * @author hankcs
 */
public class Seg
{
    public static void main(String[] args)
    {
        HanLP.Config.ShowTermNature = false;    // 关闭词性显示
        Segment segment = new CRFSegment().enableCustomDictionary(false);

        // String fileName = "/home/lzh/work/python/ccks2017/qedl/zhanghan/input.txt";
        String fileName = "/home/lzh/work/python/ccks2017/qedl/zhanghan/input_test.txt";
        String line = "";
        try{
            BufferedReader in=new BufferedReader(new FileReader(fileName));
            line=in.readLine();
            while (line!=null)
            {
                List<Term> termList = segment.seg(line);
                for(Term term : termList){
                    System.out.print(term+" ");
                }
                System.out.print("\n");
                // System.out.println(termList);
                // System.out.println(line);
                line=in.readLine();
            }
            in.close();
        }catch(IOException e){
            e.printStackTrace();
        }
    }
}