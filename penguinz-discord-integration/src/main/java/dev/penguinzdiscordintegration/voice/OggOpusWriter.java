package dev.penguinzdiscordintegration.voice;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public final class OggOpusWriter {
    private static final int SAMPLE_RATE = 48_000;
    private static final int FRAME_SIZE = 960;
    private static final int[] CRC_LOOKUP = new int[256];

    static {
        for (int i = 0; i < CRC_LOOKUP.length; i++) {
            int r = i << 24;
            for (int bit = 0; bit < 8; bit++) {
                r = (r & 0x80000000) != 0 ? (r << 1) ^ 0x04C11DB7 : r << 1;
            }
            CRC_LOOKUP[i] = r;
        }
    }

    private OggOpusWriter() {
    }

    public static byte[] write(List<byte[]> opusFrames) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        int serial = ThreadLocalRandom.current().nextInt();
        int sequence = 0;

        writePage(output, 0x02, 0L, serial, sequence++, opusHead());
        if (opusFrames.isEmpty()) {
            writePage(output, 0x04, 0L, serial, sequence, opusTags());
            return output.toByteArray();
        }

        writePage(output, 0x00, 0L, serial, sequence++, opusTags());
        long granulePosition = 0L;
        for (int i = 0; i < opusFrames.size(); i++) {
            granulePosition += FRAME_SIZE;
            int headerType = i == opusFrames.size() - 1 ? 0x04 : 0x00;
            writePage(output, headerType, granulePosition, serial, sequence++, opusFrames.get(i));
        }
        return output.toByteArray();
    }

    private static byte[] opusHead() {
        ByteBuffer buffer = ByteBuffer.allocate(19).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("OpusHead".getBytes(StandardCharsets.US_ASCII));
        buffer.put((byte) 1);
        buffer.put((byte) 1);
        buffer.putShort((short) 312);
        buffer.putInt(SAMPLE_RATE);
        buffer.putShort((short) 0);
        buffer.put((byte) 0);
        return buffer.array();
    }

    private static byte[] opusTags() {
        byte[] vendor = "Penguinz Discord Integration".getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer.allocate(8 + 4 + vendor.length + 4).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("OpusTags".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(vendor.length);
        buffer.put(vendor);
        buffer.putInt(0);
        return buffer.array();
    }

    private static void writePage(
            ByteArrayOutputStream output,
            int headerType,
            long granulePosition,
            int serial,
            int sequence,
            byte[] packet
    ) throws IOException {
        List<Integer> lacingValues = lacingValues(packet.length);
        if (lacingValues.size() > 255) {
            throw new IOException("Opus packet too large for one Ogg page");
        }

        ByteArrayOutputStream page = new ByteArrayOutputStream();
        page.write("OggS".getBytes(StandardCharsets.US_ASCII));
        page.write(0);
        page.write(headerType);
        writeLongLE(page, granulePosition);
        writeIntLE(page, serial);
        writeIntLE(page, sequence);
        writeIntLE(page, 0);
        page.write(lacingValues.size());
        for (int lacingValue : lacingValues) {
            page.write(lacingValue);
        }
        page.write(packet);

        byte[] pageBytes = page.toByteArray();
        int checksum = checksum(pageBytes);
        pageBytes[22] = (byte) checksum;
        pageBytes[23] = (byte) (checksum >>> 8);
        pageBytes[24] = (byte) (checksum >>> 16);
        pageBytes[25] = (byte) (checksum >>> 24);
        output.write(pageBytes);
    }

    private static List<Integer> lacingValues(int packetLength) {
        List<Integer> values = new ArrayList<>();
        int remaining = packetLength;
        while (remaining >= 255) {
            values.add(255);
            remaining -= 255;
        }
        values.add(remaining);
        return values;
    }

    private static int checksum(byte[] bytes) {
        int crc = 0;
        for (byte value : bytes) {
            crc = (crc << 8) ^ CRC_LOOKUP[((crc >>> 24) & 0xFF) ^ (value & 0xFF)];
        }
        return crc;
    }

    private static void writeIntLE(ByteArrayOutputStream output, int value) {
        output.write(value);
        output.write(value >>> 8);
        output.write(value >>> 16);
        output.write(value >>> 24);
    }

    private static void writeLongLE(ByteArrayOutputStream output, long value) {
        for (int i = 0; i < 8; i++) {
            output.write((int) (value >>> (8 * i)));
        }
    }
}

